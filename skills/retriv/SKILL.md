---
name: retriv
description: Adds and debugs hybrid BM25 plus vector search in TypeScript and JavaScript projects with retriv. Use when a task mentions retriv, createRetriv, retriv/db/sqlite, sqliteFts, sqliteVec, pgvector, Cloudflare Vectorize, transformersJs, autoChunker, codeChunker, rerankers, or metadata filters, or when results have tiny scores, truncated content, missing chunks, or stale chunks.
---

# retriv

retriv 0.15.0 builds keyword (SQLite FTS5 BM25), vector, or hybrid search behind one `createRetriv()` call, with optional chunking, reranking, and metadata filters. Every driver, embedding provider, chunker, and reranker is a subpath import with its own optional peer.

## Setup

Install `retriv` plus the peers of the subpaths you import. pnpm prints `ERR_PNPM_IGNORED_BUILDS` for `onnxruntime-node`; Transformers.js works without that build script.

| Import | Kind | Peers | Runtime |
|---|---|---|---|
| `retriv/db/sqlite` | hybrid in one file | `sqlite-vec` | Node 22.16+ or 24+ |
| `retriv/db/sqlite-fts` | keyword | none | Node 22.16+ or 24+ |
| `retriv/db/sqlite-vec` | vector | `sqlite-vec` | Node 22.13+ |
| `retriv/db/libsql`, `pgvector`, `upstash` | vector | `@libsql/client`, `pg`, `@upstash/vector` | Node |
| `retriv/db/cloudflare` | vector | a Vectorize binding | Workers |
| `retriv/chunkers/typescript`, `auto` | AST chunks | `typescript@^6` | Node |

Embeddings: `transformers-js` needs `@huggingface/transformers`; `openai`, `google`, `mistral`, `cohere` need `@ai-sdk/<name>` and `ai`; `ollama` needs `ollama-ai-provider-v2` and `ai`; `cloudflare-workers-ai` needs only `env.AI`. Upstash embeds server side and takes no `embeddings`.

```ts
import { readFileSync } from 'node:fs'
import { createRetriv } from 'retriv'
import { autoChunker } from 'retriv/chunkers/auto'
import sqlite from 'retriv/db/sqlite'
import { transformersJs } from 'retriv/embeddings/transformers-js'

const search = await createRetriv({
  driver: sqlite({ path: './.search/index.db', embeddings: transformersJs() }),
  chunking: autoChunker(), // ids ending .ts/.js/.mjs... use the AST chunker, the rest split as markdown
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

- `search()` rewrites the query before every driver sees it: `getUserName` becomes `get User Name getUserName`. Rerankers get the original query. To skip it, call the driver directly.
- Chunking is off until you pass `chunking`. A document that yields one chunk keeps its own id; more chunks are stored as `<id>#chunk-<n>` and carry the parent's metadata. `index()` returns the stored row count, chunks included.
- `returnMetadata` defaults to `true` on every driver.
- `categories` writes `metadata.category` into the document objects you pass, then fans each search out per category and fuses with RRF.
- `rerank` forces `returnContent` on, strips it afterwards, and fetches `limit * 3` candidates only when you set `limit`.

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
    // retriv types the binding methods as returning Promise<void>, so the workers-types binding needs a cast
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

- **TypeScript 7 breaks code chunking.** `pnpm add typescript` installs 7.x, outside the `^5 || ^6` peer range. `codeChunker` and `autoChunker` then throw `Cannot read properties of undefined (reading 'TS')`. Install `typescript@^6`. Without TypeScript, `autoChunker` throws on the first code document.
- **Node floor.** On Node 22.14 the SQLite drivers with FTS5 fail with `no such module: fts5`. Node 22.16 enabled FTS5. Before 22.13, `node:sqlite` needs a flag and retriv reports `node:sqlite not available`.
- **Scores are not similarities.** `sqlite`, composed drivers, and `categories` return RRF scores, about 0.01 to 0.05. Other drivers use their own scale, and a reranker replaces scores with its own 0 to 1 value. A driver with vectors returns up to `limit` rows even for a nonsense query. Never threshold on score.
- **`codeChunker` skips text.** Text between chunks is dropped: comments and JSDoc of the next declaration, the `export const` keyword of a variable, and statements after the last declaration. A declaration larger than `maxChunkSize` stays one chunk. Code chunks never get `range`, `scope`, or `context`; use `lineRange`. To index every line, use `markdownChunker()` for code.
- **Long text is cut by the model.** Embeddings ignore tokens past the model window (512 for the default `bge-small-en-v1.5`). Chunk long documents, or vector search only sees the start.
- **`returnMetadata: false` also removes `_chunk`.** retriv rebuilds `_chunk` from stored metadata.
- **Categories live in memory.** A new process searches unsplit until it indexes documents from two categories. Filter on `category` yourself after a restart.
- **Filters differ by driver.** In SQL drivers, `$ne` and `$nin` drop documents that lack the field, and `a.b` reads nested JSON. In Upstash and Cloudflare, `$ne` keeps them and `a.b` matches nothing. Field names allow only letters, digits, `_`, and `.`; `my-field` throws.
- **One model per index.** The vector table takes its dimensions from the first open. Another model on the same file throws `Dimension mismatch`. `cachedEmbeddings` hashes only the text, so give each model its own storage.
- **Do not pass `dimensions` to `transformersJs()` for a known model.** A wrong value slices the output into silent garbage vectors.
- **pgvector builds an empty ivfflat index.** The driver creates it when it opens, before rows exist. Searches then return far fewer rows than `limit`, and filtered searches often return none. After the first load, run `REINDEX INDEX <table>_embedding_idx` (default `vectors_embedding_idx`).
- **Cloudflare driver:** `remove()` with chunking and `clear()` throw. Each vector stores its text in metadata. Cloudflare documents a 10 KiB metadata limit per vector, so chunk large documents. `retriv/embeddings/cached` imports `node:crypto` and needs `nodejs_compat`.
- **Google embeddings read `GOOGLE_GENERATIVE_AI_API_KEY`**, not `GOOGLE_API_KEY` as the option comment says.
- `remove`, `listIds`, `clear`, and `close` are optional on the returned type; call them with `?.`.

## Version limits

| Old call | New call | Since |
|---|---|---|
| chunking on by default, `chunking: { chunker }` | opt-in `chunking: markdownChunker()` | 0.7.0 |
| `retriv/chunkers/code` with `code-chunk` | `retriv/chunkers/typescript` with `typescript` | 0.8.1 |
| `retriv/embeddings/transformers`, `transformers()` | `retriv/embeddings/transformers-js`, `transformersJs()` | 0.1.0 |

## Config

Options for each driver, chunker, embedding provider, and reranker are typed on its factory. The [README](https://github.com/skilld-dev/retriv#readme) lists them.
