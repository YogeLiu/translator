import os
import shutil
import json
from huggingface_hub import login, create_repo, upload_folder

# ========== Step 1: 登录 ==========
# 推荐：将token保存为环境变量 HUGGINGFACE_TOKEN
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise ValueError("请将 Hugging Face token 设置为环境变量 HUGGINGFACE_TOKEN")

login(token=HF_TOKEN)

# ========== Step 2: 加载配置 ==========
from config import Config

config = Config()

# ========== Step 3: 创建上传临时目录 ==========
UPLOAD_DIR = "hf_upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 复制模型
model_src = "model/best_model.pth"
model_dst = os.path.join(UPLOAD_DIR, "best_model.pth")
shutil.copy(model_src, model_dst)

# 保存 config.json
config_dict = config.__dict__
with open(os.path.join(UPLOAD_DIR, "config.json"), "w") as f:
    json.dump(config_dict, f, indent=4)

# 复制 tokenizer 文件（假设是 sentencepiece）
src_tokenizer_src = "tokenizer/src_tokenizer.model"
tgt_tokenizer_src = "tokenizer/tgt_tokenizer.model"
src_tokenizer_dst = os.path.join(UPLOAD_DIR, "src_tokenizer.model")
tgt_tokenizer_dst = os.path.join(UPLOAD_DIR, "tgt_tokenizer.model")
shutil.copy(src_tokenizer_src, src_tokenizer_dst)
shutil.copy(tgt_tokenizer_src, tgt_tokenizer_dst)


# ========== Step 4: 上传到 Hugging Face ==========
REPO_ID = "YogeLiu/zh-en-translater"  # 修改为你的用户名/模型名
PRIVATE = False

# 创建 repo（如果已存在可跳过或捕获异常）
create_repo(repo_id=REPO_ID, private=PRIVATE, exist_ok=True)

# 上传文件夹
upload_folder(
    repo_id=REPO_ID,
    folder_path=UPLOAD_DIR,
    path_in_repo=".",  # 上传到根目录
    commit_message="Initial model upload",
)

print(f"✅ 模型已成功上传到 Hugging Face Hub: https://huggingface.co/{REPO_ID}")
