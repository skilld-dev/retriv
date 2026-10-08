---
name: retriv
description: Adds and debugs hybrid BM25 plus vector search in TypeScript and JavaScript projects with retriv. Use when a task mentions retriv, createRetriv, retriv/db/sqlite, sqliteFts, sqliteVec, pgvector, Cloudflare Vectorize, transformersJs, autoChunker, codeChunker, rerankers, or metadata filters, or when results have tiny scores, truncated content, missing chunks, or stale chunks.
---

# retriv

retriv 0.15.0 builds keyword (SQLite FTS5 BM25), vector, or hybrid search behind one `createRetriv()` call, with optional chunking, reranking, and metadata filters. Each driver, embedder, chunker, and reranker is a subpath import with an optional peer.

## Setup

Install `retriv` and the peers of the subpaths you import. With pnpm 12, adding `@huggingface/transformers` exits 1 with `ERR_PNPM_IGNORED_BUILDS` (`onnxruntime-node`, `protobufjs`), but the install works. To silence it, set both to `false` under `allowBuilds` in `pnpm-workspace.yaml`.

| Import | Provides | Peers | Runtime |
|---|---|---|---|
| `retriv/db/sqlite` | hybrid in one file | `sqlite-vec` | Node 22.16+ or 24+ |
| `retriv/db/sqlite-fts` | keyword | none | Node 22.16+ or 24+ |
| `retriv/db/sqlite-vec` | vector | `sqlite-vec` | Node 22.13+ |
| `retriv/db/libsql`, `pgvector`, `upstash` | vector | `@libsql/client`, `pg`, `@upstash/vector` | Node |
| `retriv/db/cloudflare` | vector | a Vectorize binding | Workers |
| `retriv/chunkers/typescript`, `auto` | AST chunks | `typescript@^6` | Node |
| `retriv/rerankers/transformers-js` | `crossEncoder()` | `@huggingface/transformers` | Node |
| `retriv/rerankers/cohere`, `jina` | `cohereReranker()`, `jinaReranker()` | `apiKey`, or `COHERE_API_KEY`, `JINA_API_KEY` | Node |

Embeddings: `transformers-js` needs `@huggingface/transformers`; `openai`, `google`, `mistral`, `cohere` need `@ai-sdk/<name>` and `ai`; `ollama` needs `ollama-ai-provider-v2` and `ai`.

```ts
import { readFileSync } from 'node:fs'
import { createRetriv } from 'retriv'
import { autoChunker } from 'retriv/chunkers/auto'
import sqlite from 'retriv/db/sqlite'
import { transformersJs } from 'retriv/embeddings/transformers-js'

const search = await createRetriv({
  driver: sqlite({ path: './.search/index.db', embeddings: transformersJs() }),
  chunking: autoChunker(), // .ts and .js ids use the AST chunker; the rest split as markdown
})

await search.index([
  { id: 'src/auth.ts', content: readFileSync('src/auth.ts', 'utf8'), metadata: { type: 'code' } },
  { id: 'docs/guide.md', content: readFileSync('docs/guide.md', 'utf8'), metadata: { type: 'docs' } },
])

const results = await search.search('password hashing', { limit: 5, filter: { type: 'code' } })
for (const r of results)
  console.log(r.id, r._chunk?.parentId, r._chunk?.lineRange)
await search.close?.()
```

## Automatic behaviour

- `search()` rewrites the query for every driver: `getUserName` becomes `get User Name getUserName`. Rerankers get the original query.
- Chunking is off until you pass `chunking`. A document with one chunk keeps its id; more chunks are stored as `<id>#chunk-<n>` with the parent's metadata. `index()` resolves to `{ count }`, chunks included. Without chunking, the default `bge-small-en-v1.5` embeds only the first 512 tokens.
- `returnMetadata` defaults to `true`; `false` also removes `_chunk`.
- `categories` writes `metadata.category` into the documents you pass, then searches each category and fuses with RRF. The seen categories live in memory: after a restart, search is unsplit until the process indexes two categories.
- `rerank` forces `returnContent` on, strips it afterwards, and fetches `limit * 3` candidates only when you set `limit`. Rerankers score the 5-line snippet, so pair `rerank` with small chunks.

## Common tasks

`returnContent: true` returns a 5-line snippet around the best matching line, not the stored text. Keep your own copy and re-index changed documents through `remove()`:

```ts
import type { Document } from 'retriv'
import { createRetriv } from 'retriv'
import { markdownChunker } from 'retriv/chunkers/markdown'
import sqliteFts from 'retriv/db/sqlite-fts'

const docs = new Map<string, Document>()
const search = await createRetriv({ driver: sqliteFts({ path: './.search/fts.db' }), chunking: markdownChunker() })

export async function upsert(doc: Document): Promise<void> {
  await search.remove?.([doc.id]) // index() alone leaves old <id>#chunk-<n> rows behind
  await search.index([doc])
  docs.set(doc.id, doc)
}

export async function find(query: string): Promise<{ id: string, text: string }[]> {
  const results = await search.search(query, { limit: 5 })
  return results.map((r) => {
    const parent = docs.get(r._chunk?.parentId ?? r.id)
    const [start, end] = r._chunk?.lineRange ?? [1, Infinity]
    return { id: r.id, text: parent?.content.split('\n').slice(start - 1, end).join('\n') ?? '' }
  })
}
```

Cloudflare Worker with Vectorize and Workers AI. Create the index with 384 dimensions and the cosine metric for the default `@cf/baai/bge-small-en-v1.5`:

```ts
import type { CloudflareConfig } from 'retriv/db/cloudflare'
import { createRetriv } from 'retriv'
import { markdownChunker } from 'retriv/chunkers/markdown'
import cloudflare from 'retriv/db/cloudflare'
import { cloudflareWorkersAi } from 'retriv/embeddings/cloudflare-workers-ai'

interface Env { VECTORIZE: Vectorize, AI: Ai }

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // retriv types these methods as Promise<void>, so the workers-types binding needs a cast
    const binding = env.VECTORIZE as unknown as CloudflareConfig['binding']
    const search = await createRetriv({
      driver: cloudflare({ binding, embeddings: cloudflareWorkersAi({ ai: env.AI }) }),
      chunking: markdownChunker(),
    })
    const query = new URL(request.url).searchParams.get('q') ?? ''
    return Response.json(await search.search(query, { limit: 5 }))
  },
}
```

## Traps

- **TypeScript 7 breaks code chunking.** `pnpm add typescript` installs 7.x, outside the peer range, and both code chunkers throw `Cannot read properties of undefined (reading 'TS')`. Install `typescript@^6`. Without TypeScript, `autoChunker` throws on the first code document.
- **Node floor.** Below the Node versions in the table, SQLite drivers fail with `no such module: fts5` (22.14, 22.15) or `node:sqlite not available` (before 22.13).
- **Scores are not similarities.** `sqlite`, composed drivers, `categories`, and multi-token `sqlite-fts` queries return RRF scores, about 0.005 to 0.05; `search()` makes `getUserName` multi-token. A single-token `sqlite-fts` hit scores near 1; a reranker sets 0 to 1. A driver with vectors returns up to `limit` rows even for a nonsense query. Never threshold on score.
- **`codeChunker` skips text.** At each chunk boundary it drops the next declaration's comments and JSDoc, and a variable's `export const`. It also drops statements after the last declaration. A declaration over `maxChunkSize` stays one chunk, and code chunks get no `range`, `scope`, or `context`. To index every line, chunk code with `markdownChunker()`.
- **Filters differ by driver.** SQL drivers drop documents that lack the field on `$ne` and `$nin`, and throw on a field name outside letters, digits, `_`, and `.`. The SQLite and libsql drivers read `a.b` as nested JSON. pgvector reads `a.b` as one literal key and never matches a boolean; flatten nested fields and store booleans as strings.
- **Upstash and Cloudflare filter in memory** over the `limit * 4` nearest vectors, so a selective filter can return few rows or none; raise `limit`. There, `$ne` keeps a document without the field but drops one without metadata, `a.b` never matches, and a bad field name returns `[]`.
- **One model per index.** The vector table takes its dimensions from the first open; another model on the same file throws `Dimension mismatch`. `cachedEmbeddings` hashes only the text, so give each model its own storage. Never pass `dimensions` to `transformersJs()` for a known model; a wrong value gives silent garbage vectors.
- **pgvector recall is low by default.** The driver builds ivfflat (`lists = 100`) before rows exist, and `ivfflat.probes` is 1: recall@10 is near 0.2, and about 0.44 after `REINDEX`. Replace it with HNSW under the same name, which the driver's `CREATE INDEX IF NOT EXISTS` keeps: `DROP INDEX vectors_embedding_idx; CREATE INDEX vectors_embedding_idx ON vectors USING hnsw (embedding vector_cosine_ops)`. Use `<table>_embedding_idx` and the opclass for `metric`. To keep ivfflat, run `REINDEX`, then `ALTER DATABASE <db> SET ivfflat.probes = 10`.
- **Cloudflare:** `remove()` with chunking and `clear()` throw. Each vector keeps its text in metadata, which Cloudflare caps at 10 KiB, so chunk large documents. `retriv/embeddings/cached` needs `nodejs_compat`.

## Version limits

| Old call | New call | Since |
|---|---|---|
| chunking on by default, `chunking: { chunker }` | opt-in `chunking: markdownChunker()` | 0.7.0 |
| `retriv/chunkers/code` with `code-chunk` | `retriv/chunkers/typescript` with `typescript` | 0.8.1 |

## Config

Factory options are typed in `node_modules/retriv/dist/<subpath>.d.mts`, such as `dist/db/pgvector.d.mts`. The README does not list them all.
