; Structural reference captures for Kotlin. Roots marked @reference.* are
; forwarded to _kotlin_records, which re-dispatches on node.type;
; sub-captures are decorative.

; --- imports: one row per header (plain and wildcard) -----------------------

(import_header) @reference.import

; --- calls: plain identifier calls and navigation calls --------------------

(call_expression) @reference.call

; --- non-call member access through navigation expressions ------------------

(navigation_expression) @reference.member_access
(directly_assignable_expression) @reference.member_access

; --- heritage: delegation specifiers are inheritance edges ------------------
; `class A : B(), C` -- one capture per supertype; the handler takes the head
; type as `inheritance` and any type arguments as `type_use`.

(delegation_specifier) @reference.heritage

; --- type expressions (handler descends to the naming leaves) --------------
; `x: Foo`, `: Bar`, `val v: Foo` -- user_type owns every named type spelling,
; so the handler emits the `type_use` rows and the identifier fallback cuts
; the parallel plain reads (`handler_owned_type_parents`).

(user_type) @reference.type_use

; --- declaration parameters -------------------------------------------------

(function_declaration
  (function_value_parameters) @declaration.parameters)

; --- exports: public top-level declarations (public is Kotlin's default) ---

(source_file (class_declaration) @reference.export)
(source_file (object_declaration) @reference.export)
(source_file (function_declaration) @reference.export)
(source_file (property_declaration) @reference.export)
(source_file (type_alias) @reference.export)

; --- identifier fallback (bindings excluded in _identifier_record) -----------

(simple_identifier) @reference.identifier
(type_identifier) @reference.identifier
