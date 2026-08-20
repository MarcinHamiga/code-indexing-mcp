# code-indexing-mcp

Local Tree-sitter code indexing with hybrid semantic and full-text search,
served over the Model Context Protocol. This is the TypeScript port of
[the Python build](https://github.com/MarcinHamiga/code-indexing-mcp); the
two share on-disk state, so an existing index opens with either.

**This package is published on the `next` dist-tag** while the Python build
remains the default; the migration plan's cutover notes live in the
repository under `docs/plans/`.

## Requirements

The binaries execute TypeScript sources directly, so this package requires
[Bun](https://bun.sh) 1.2 or newer on the embedding path (`engines.bun`
declares it; npm itself will not check it). Every other runtime — Node
included — can talk to the stdio MCP server it starts.

## Install

```sh
bun add -g code-indexing-mcp@next
code-indexing-mcp init /path/to/repo
code-indexing-mcp index /path/to/repo
code-indexing-mcp serve
```

The first index run downloads the `jinaai/jina-embeddings-v2-base-code`
ONNX model once. Embedding runs on the CPU through `onnxruntime-node`;
accelerator promotion (CUDA, DirectML, WebGPU, CoreML) is wired but has not
been promoted on real hardware yet, so those providers fall back to CPU.

## One platform difference

Windows omits GDShader indexing (`.gdshader` / `.gdshaderinc`): the only
npm source for that grammar cannot load its Windows binding. GDScript and
Godot resources are unaffected, as is every other platform.

## More

The repository README covers the daemon, the installer, MCP client
configuration, and the embedding backends:
<https://github.com/MarcinHamiga/code-indexing-mcp>.
