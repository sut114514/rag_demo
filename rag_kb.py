"""PDF 配方知识库：按配方切分、中文语义向量化、持久化及余弦检索。"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import re
import sys

import numpy as np
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
DEFAULT_PDF = ROOT / 'output/terraria/泰拉瑞亚30条饰品合成路径.pdf'
DEFAULT_KB = ROOT / 'knowledge_base/terraria'
MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
CACHE_DIR = ROOT / '.cache/fastembed'
RECORD_START = re.compile(r'(?m)^\s*(\d{2})\.\s*([^\n]+)')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_records(pdf_path, expected_count=30):
    """直接从 PDF 取文本，按编号切分，保留可跨页的完整记录及页码。"""
    reader = PdfReader(pdf_path)
    pages = [(page.extract_text() or '').replace('\r\n', '\n') for page in reader.pages]
    if not any(page.strip() for page in pages):
        raise ValueError('PDF 没有可提取文本，需要先进行 OCR。')
    full_text = '\n'.join(pages)
    offsets = np.cumsum([0] + [len(page) + 1 for page in pages])
    matches = list(RECORD_START.finditer(full_text))
    if len(matches) != expected_count:
        raise ValueError(f'预期 {expected_count} 条配方，实际解析 {len(matches)} 条；停止入库。')
    source_hash = sha256(pdf_path)
    records = []
    for i, match in enumerate(matches):
        start = match.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(full_text)
        raw = full_text[start:end].strip()
        # PDF 的自动换行合并到同一字段，避免配方、URL 被截断。
        text = re.sub(r'\n(?!\s*(?:游戏版本|合成配方|制作站|说明|来源)：)', '', raw)
        text = '\n'.join(line.strip() for line in text.splitlines())
        fields = {}
        for line in text.splitlines()[1:]:
            key, sep, value = line.partition('：')
            if sep:
                fields[key] = value.strip()
        for required in ('游戏版本', '合成配方', '制作站', '来源'):
            if not fields.get(required):
                raise ValueError(f'配方 {match.group(1)} 缺少字段：{required}')
        number = int(match.group(1))
        title = match.group(2).strip()
        records.append({
            'id': f'terraria-{number:02d}',
            'recipe_number': number,
            'title': title,
            'game_version': fields['游戏版本'],
            'recipe': fields['合成配方'],
            'crafting_station': fields['制作站'],
            'notes': fields.get('说明', ''),
            'source_url': fields['来源'],
            'source_pdf': str(pdf_path.resolve()),
            'source_sha256': source_hash,
            'page_start': int(np.searchsorted(offsets, start, side='right')),
            'page_end': int(np.searchsorted(offsets, start + len(raw) - 1, side='right')),
            'text': text,
            'embedding_text': '\n'.join(
                [f'泰拉瑞亚饰品：{title}', f'合成配方：{fields["合成配方"]}',
                 f'制作站：{fields["制作站"]}']
                + ([f'说明：{fields["说明"]}'] if fields.get('说明') else [])
            ),
        })
    if [r['recipe_number'] for r in records] != list(range(1, expected_count + 1)):
        raise ValueError('配方编号重复、缺失或顺序异常；停止入库。')
    return records, len(pages)


def load_model(model_name=MODEL_NAME, allow_download=False):
    # 查询默认离线；仅显式构建时允许下载公开模型。文档不上传。
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model_name, cache_dir=str(CACHE_DIR),
                         threads=4, local_files_only=not allow_download)


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError('嵌入向量包含非有限值或零向量。')
    return vectors / norms


def build(pdf_path, kb_dir, allow_download=False, expected_count=30):
    records, page_count = extract_records(pdf_path, expected_count)
    print(f'已从 PDF 的 {page_count} 页提取 {len(records)} 条完整配方。', flush=True)
    model = load_model(allow_download=allow_download)
    # 来源 URL、重复版本号保存在元数据中，不参与语义向量化。
    vectors = normalize(list(model.passage_embed([r['embedding_text'] for r in records])))
    if vectors.ndim != 2 or vectors.shape[0] != len(records):
        raise ValueError('向量数与配方数不一致。')
    kb_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = kb_dir / 'chunks.jsonl'
    chunks_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    vectors_path = kb_dir / 'vectors.npy'
    np.save(vectors_path, vectors, allow_pickle=False)
    manifest = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_pdf': str(pdf_path.resolve()),
        'source_sha256': sha256(pdf_path),
        'page_count': page_count,
        'chunk_count': len(records),
        'chunk_strategy': 'one_complete_recipe_per_chunk',
        'embedding_model': MODEL_NAME,
        'vector_dimension': int(vectors.shape[1]),
        'vector_dtype': str(vectors.dtype),
        'normalized': True,
        'similarity': 'cosine',
        'index_type': 'numpy_flat_exact',
        'chunks_sha256': sha256(chunks_path),
        'vectors_sha256': sha256(vectors_path),
        'packages': {p: version(p) for p in ('fastembed', 'onnxruntime', 'pypdf', 'numpy')},
    }
    (kb_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'知识库已保存：{kb_dir}\n向量矩阵：{vectors.shape}，float32，余弦相似度。')
    return manifest


class KnowledgeBase:
    def __init__(self, kb_dir=DEFAULT_KB):
        kb_dir = Path(kb_dir)
        self.manifest = json.loads((kb_dir / 'manifest.json').read_text(encoding='utf-8'))
        for filename, hash_key in [('chunks.jsonl', 'chunks_sha256'), ('vectors.npy', 'vectors_sha256')]:
            if sha256(kb_dir / filename) != self.manifest[hash_key]:
                raise ValueError(f'{filename} 校验失败，请重新构建。')
        self.records = [json.loads(line) for line in (kb_dir / 'chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        self.vectors = np.load(kb_dir / 'vectors.npy', allow_pickle=False)
        expected_shape = (self.manifest['chunk_count'], self.manifest['vector_dimension'])
        if len(self.records) != expected_shape[0] or self.vectors.shape != expected_shape:
            raise ValueError('知识库记录数量、向量矩阵与清单不一致。')
        if not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise ValueError('知识库向量不是有效的单位向量。')
        self._model = None

    def search(self, question, top_k=3):
        if not question.strip():
            raise ValueError('问题不能为空。')
        if top_k < 1:
            raise ValueError('top_k 必须大于 0。')
        if self._model is None:
            self._model = load_model(self.manifest['embedding_model'])
        query = normalize(next(iter(self._model.query_embed(question))))
        scores = self.vectors @ query
        indices = np.argsort(-scores, kind='stable')[:top_k]
        return [dict(self.records[int(i)], score=float(scores[i])) for i in indices]


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kb-dir', type=Path, default=DEFAULT_KB)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('build', help='从 PDF 构建知识库')
    create.add_argument('--pdf', type=Path, default=DEFAULT_PDF)
    create.add_argument('--allow-download', action='store_true')
    create.add_argument('--expected-count', type=int, default=30)
    query = sub.add_parser('search', help='语义向量检索')
    query.add_argument('question')
    query.add_argument('--top-k', type=int, default=3)
    query.add_argument('--json', action='store_true', dest='as_json')
    sub.add_parser('info', help='校验并显示知识库信息')
    args = parser.parse_args()
    try:
        if args.command == 'build':
            build(args.pdf, args.kb_dir, args.allow_download, args.expected_count)
        else:
            kb = KnowledgeBase(args.kb_dir)
            if args.command == 'info':
                print(json.dumps(kb.manifest, ensure_ascii=False, indent=2))
            else:
                results = kb.search(args.question, args.top_k)
                if args.as_json:
                    print(json.dumps(results, ensure_ascii=False, indent=2))
                else:
                    for rank, result in enumerate(results, 1):
                        print(f'[{rank}] {result["title"]} | 余弦相似度 {result["score"]:.4f} | PDF 第 {result["page_start"]} 页')
                        print(result['text'] + '\n')
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
