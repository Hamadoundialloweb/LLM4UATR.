import os
import torch
import torch.nn as nn
from mamba_ssm import Mamba, Mamba2
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

from beats_wrapper import BEaTsFeatureExtractor


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# ========== GPT-2 Config ==========
gpt2_path = "/remote-home/share/dmb_nas/jxd/language_model/openai-community/gpt2"

gpt2_config = GPT2Config.from_pretrained(gpt2_path)
gpt2_config.num_hidden_layers = 2
gpt2_config.output_attentions = True
gpt2_config.output_hidden_states = True

gpt2_model = GPT2Model.from_pretrained(
    gpt2_path,
    trust_remote_code=True,
    local_files_only=True,
    config=gpt2_config,
).to("cuda")
gpt2_model.eval()

gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
    gpt2_path,
    trust_remote_code=True,
    local_files_only=True
)

if gpt2_tokenizer.eos_token:
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
else:
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    gpt2_tokenizer.pad_token = '[PAD]'


# ========== Configuration ==========
class Config:
    d_model = 128
    d_ff = 512
    llm_dim = 768
    beats_dim = 768
    description = (
        "The Shipsear dataset contains nine ship classes: Dredger, Fishboat, "
        "Motorboat, Mussel boat, Natural ambient noise, Ocean liner, "
        "Passengers, RORO, Sailboat. Task: underwater acoustic classification. "
        "The Deepship dataset contains four ship classes: Cargo, Tanker, Tug, Passengership."
        "The Oceanship dataset contains six ship classes: Cargo, Fishing, Tug, Passenger, Pleasure Craft, Towing ."
    )


configs = Config()


# ========== Modèle ==========
class MNISTMambaModelWithBEaTS(nn.Module):
    def __init__(
        self,
        classes: int,
        use_beats: bool = False,
        beats_checkpoint: str = "/remote-home/Diallo/Beat/BEATs_iter3_plus_AS2M.pt",
        beats_frozen: bool = True,
        fusion_strategy: str = None,
        configs=configs
    ):
        super().__init__()

        self.use_beats = use_beats
        self.configs = configs
        self.num_classes = classes
        self.description = configs.description
        self.fusion_strategy = fusion_strategy

        print(f"\n🏗️ Initialisation modèle")
        print(f"   Classes: {classes}")
        print(f"   Use BEATs: {use_beats}")

        # ===== BEATs =====
        if self.use_beats:
            self.beats_extractor = BEaTsFeatureExtractor(
                checkpoint_path=beats_checkpoint,
                target_sr=16000,
                freeze=beats_frozen
            )

            self.beats_proj = nn.Linear(self.beats_extractor.hidden_dim, configs.d_model)
            self.beats_ln = nn.LayerNorm(configs.d_model)

            self.beats_ffn1 = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(configs.d_ff)
            )

            self.beats_ffn2 = nn.Sequential(
                nn.Linear(configs.d_ff, configs.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(configs.d_model)
            )

            self.alpha = nn.Parameter(torch.tensor(1.0))

        # ===== Spectral =====
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.3)
        )

        self.spectral_proj = None

        # ===== Mamba =====
        self.mamba1 = Mamba(d_model=configs.d_model, d_state=8, d_conv=4, expand=4)
        self.mamba2 = Mamba2(d_model=configs.d_model, d_state=8, d_conv=4, expand=4)

        # ===== GPT / LLM projection =====
        self.llm_in = nn.Linear(configs.d_model, configs.llm_dim)
        self.llm_out = nn.Linear(configs.llm_dim, configs.d_model)

        # ===== CROSS-ATTENTION =====
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=configs.d_model,
            num_heads=4,
            batch_first=True
        )
        self.cross_ln = nn.LayerNorm(configs.d_model)

        # ===== Classifier =====
        self.classifier_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(configs.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, classes)
        )

        print("✅ Modèle prêt avec Cross-Attention\n")

    def _process_spectral(self, x):
        x = self.spectral_conv(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if x.size(-1) != self.configs.d_model:
            if self.spectral_proj is None:
                self.spectral_proj = nn.Linear(x.size(-1), self.configs.d_model).to(x.device)
            x = self.spectral_proj(x)

        return x

    def _process_beats(self, x):
        x = self.beats_proj(x)
        x = self.beats_ln(x)

        e0 = x.clone()
        eprime = self.beats_ffn1(x)
        eprime = self.beats_ffn2(eprime)

        alpha = torch.sigmoid(self.alpha)
        x = alpha * e0 + (1 - alpha) * eprime
        return x

    def forward(self, x_main):
        device = x_main.device
        B = x_main.size(0)

        # ===== Input =====
        if self.use_beats:
            x = self._process_beats(x_main)
        else:
            x = self._process_spectral(x_main)

        # ===== Mamba =====
        x_fwd = self.mamba1(x)
        x_bwd = self.mamba2(x.flip([1])).flip([1])
        x = x_fwd + x_bwd

        # ===== GPT-2 =====
        prompt = gpt2_tokenizer(
            self.description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).input_ids.to(device)

        prompt_emb = gpt2_model.get_input_embeddings()(prompt)
        prompt_emb = prompt_emb.repeat(B, 1, 1)

        combined = prompt_emb
        max_len = gpt2_model.config.n_positions
        if combined.size(1) > max_len:
            combined = combined[:, :max_len, :]

        with torch.no_grad():
            gpt_out = gpt2_model(inputs_embeds=combined).last_hidden_state

        x_gpt = self.llm_out(gpt_out)

        # ===== Align lengths =====
        L = min(x.size(1), x_gpt.size(1))
        x = x[:, :L, :]
        x_gpt = x_gpt[:, :L, :]

        # ===== CROSS-ATTENTION =====
        attn_output, _ = self.cross_attn(
            query=x_gpt,
            key=x,
            value=x,
            need_weights=False
        )

        x_cross = self.cross_ln(x_gpt + attn_output)

        # ===== Classification =====
        x_cls = x_cross.permute(0, 2, 1)
        logits = self.classifier_proj(x_cls)

        return logits
