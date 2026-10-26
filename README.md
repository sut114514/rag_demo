# 泰拉瑞亚 RAG 知识库

从 `output/terraria/泰拉瑞亚30条饰品合成路径.pdf` 提取 30 条配方，每条配方独立作为一个片段。保留材料、数量、制作站、版本说明、来源链接及 PDF 页码。

## Python 环境

项目虚拟环境为 `ragdemo`，使用 Python 3.13。后续依赖均安装到这个环境。

PowerShell 激活：

```powershell
.\ragdemo\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

也可以直接调用环境中的 Python，无需激活或修改执行策略：

```powershell
.\ragdemo\Scripts\python.exe -m pip install -r requirements.txt
```

完整依赖版本见 `requirements.lock.txt`，可用 `-r requirements.lock.txt` 恢复本次运行环境。

## 已建立的知识库

- 中文嵌入模型：`BAAI/bge-small-zh-v1.5`，通过 FastEmbed / ONNX 在 CPU 本地运行。
- 每条配方一个片段，512 维 float32 向量，L2 归一化。
- `knowledge_base/terraria/vectors.npy`：向量矩阵。
- `knowledge_base/terraria/chunks.jsonl`：原文、向量化文本和来源元数据，每行一条配方。
- `knowledge_base/terraria/manifest.json`：模型、维度、PDF 哈希、索引文件哈希及依赖版本。
- `.cache/fastembed`：本地模型文件。保留此目录即可离线重新构建和查询。

30 条数据使用 NumPy 精确余弦检索，无需额外启动数据库服务。检索分数是相似度，不是正确率或置信度。网页问答通过 DeepSeek API，根据检索到的片段生成回答。

## 网页问答

项目根目录的 `.env` 配置 DeepSeek API 密钥、接口地址、模型和端口。`.env` 已加入 Git 忽略规则，密钥仅由后端读取，不会返回给浏览器。环境变量优先于 `.env`。

```powershell
.\ragdemo\Scripts\python.exe app.py
```

也可以直接双击项目根目录的 `start.bat`，或执行 `start.ps1`。BAT 会自动进入项目目录，直接调用 `ragdemo` 环境中的 Python，无需手动激活环境，并在服务成功启动后自动使用 Microsoft Edge 打开 http://127.0.0.1:8000 （如修改 `.env` 中的 `PORT`，则打开对应端口）。保持终端窗口运行，按 Ctrl+C 停止。默认仅监听本机。手动运行 `app.py` 时添加 `--open-edge` 也可自动打开 Edge。

页面支持连续追问、新对话、示例问题及展开查看配方来源。点击来源可打开 PDF 对应页码。每次提问先本地向量检索，再补齐中间饰品的依赖配方，将问题、最近四轮历史及参考片段发送给 DeepSeek。对缺失资料的内容要求明确说明，不补编配方。

默认模型为 `deepseek-flash`，可通过 `.env` 的 `DEEPSEEK_MODEL` 修改。修改密钥或模型后需重启服务。401、余额不足、限流、网络失败和超时会在页面显示对应错误，不会伪装成成功回答。

后端入口为 `app.py`，前端为 `static/`。`GET /api/health` 返回配置和知识库状态（不包含密钥，也不代表已验证 API 连通）；`POST /api/chat` 提交 `question`、`history`、`top_k`，返回 `answer` 和 `sources`。网页仅提供本地 demo，未添加账号认证。

## 检索

```powershell
.\ragdemo\Scripts\python.exe rag_kb.py search "泰拉闪耀靴怎么合成？" --top-k 3
.\ragdemo\Scripts\python.exe rag_kb.py search "十字章护身符需要什么材料？" --json
.\ragdemo\Scripts\python.exe rag_kb.py info
```

Python 调用：

```python
from rag_kb import KnowledgeBase

kb = KnowledgeBase()
hits = kb.search("魔法手铐怎么做？", top_k=3)
context = "\n\n".join(hit["text"] for hit in hits)
# 将 context 与用户问题一起传给后续接入的大语言模型。
```

## 从 PDF 重新构建

```powershell
.\ragdemo\Scripts\python.exe rag_kb.py build
```

如果本地尚未缓存模型，首次构建允许下载：

```powershell
.\ragdemo\Scripts\python.exe rag_kb.py build --allow-download
```

默认构建和所有查询只加载本地模型。PDF 解析与向量化均在本地执行。构建脚本严格检查 30 条配方编号及必填字段，不会在解析缺失时生成部分索引。该解析器针对本项目的编号配方 PDF；其他布局的 PDF 需要调整切分规则。

## 验证

```powershell
.\ragdemo\Scripts\python.exe -m unittest -v test_rag_kb.py test_app.py
```

验证 PDF 提取内容与审核后的 TXT 一致、页码与版本说明完整，并用 7 个中文问题检查前三条检索结果是否包含对应配方。这组样例用于基本功能验证，不代表通用检索准确率。
