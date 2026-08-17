/**
 * Installer package -- scaffolding only until Phase 8.
 *
 * The split mirrors today's Python layout, where the installer is a separate
 * package that the serving environment never imports. Keeping the boundary
 * from Phase 0 means the server package cannot accidentally grow a dependency
 * on the wizard, which is what makes `bun build --compile` (decision D5) a
 * question about one package rather than the whole tree.
 */

/** Marker export so the package has a typed entry point before Phase 8 fills it in. */
export const PHASE: "scaffolding" = "scaffolding";
