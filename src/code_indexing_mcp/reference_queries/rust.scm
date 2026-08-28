; Structural reference captures for Rust. Roots marked @reference.* are forwarded
; to _rust_records, which re-dispatches on node.type; sub-captures are decorative.

; --- imports: one row set per use_declaration (plain, aliased, grouped, glob)

(use_declaration) @reference.use_declaration

; --- calls: plain, path-qualified, method, and turbofish --------------------

(call_expression) @reference.call

; --- non-call member access (reads and writes through fields) ---------------

(field_expression) @reference.field_expression

; --- impl blocks: the implemented trait is an inheritance edge and the ------
; --- self type a type use; methods qualify through _symbol_context ----------

(impl_item) @reference.impl_item

; --- struct literals name their type ----------------------------------------

(struct_expression) @reference.struct_expression

; --- type-bearing declarations (handler descends the type fields) -----------

(function_item) @reference.function_item
(function_signature_item) @reference.function_signature_item
(let_declaration) @reference.let_declaration
(field_declaration) @reference.field_declaration
(parameter) @reference.parameter
(enum_variant) @reference.enum_variant
(const_item) @reference.const_item
(static_item) @reference.static_item

; --- declaration parameters (definition shapes; self lands in slot 0) -------

(function_item parameters: (parameters) @declaration.parameters)

(function_signature_item parameters: (parameters) @declaration.parameters)

; --- exports: top-level items (the handler checks `pub` visibility) ---------
; The capture must sit on the source_file parent so nested impl methods and
; module-private items never export.

(source_file (struct_item) @reference.export)
(source_file (enum_item) @reference.export)
(source_file (trait_item) @reference.export)
(source_file (function_item) @reference.export)
(source_file (const_item) @reference.export)
(source_file (static_item) @reference.export)
(source_file (use_declaration) @reference.export)

; --- identifier fallback (bindings excluded in _identifier_record) ----------

(identifier) @reference.identifier
(type_identifier) @reference.identifier
(field_identifier) @reference.identifier
