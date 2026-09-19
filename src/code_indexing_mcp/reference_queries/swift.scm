; Structural reference captures for Swift. Roots marked @reference.* are
; forwarded to _swift_records, which re-dispatches on node.type;
; sub-captures are decorative.

; --- imports: one row per module import -------------------------------------

(import_declaration) @reference.import

; --- calls: plain identifier calls and navigation calls --------------------

(call_expression) @reference.call

; --- non-call member access through navigation expressions ------------------

(navigation_expression) @reference.member_access

; --- heritage: inheritance clauses are inheritance edges -------------------
; `class C: D, E` / `protocol P: Q` -- one capture per listed supertype; the
; handler takes the head type as `inheritance` and any type arguments as
; `type_use`.

(inheritance_specifier) @reference.heritage

; --- type expressions (handler descends to the naming leaves) --------------
; `x: Foo`, `-> Bar`, `var v: Foo` -- user_type owns every named type
; spelling, so the handler emits the `type_use` rows and the identifier
; fallback cuts the parallel plain reads (`handler_owned_type_parents`).

(user_type) @reference.type_use

; --- declaration parameters (Swift names no parameter-list wrapper, so each
; bare `parameter` is captured and _one_parameter_list treats one as a
; single slot; positions renumber over the merged lists) --------------------

(function_declaration
  (parameter) @declaration.parameters)

(init_declaration
  (parameter) @declaration.parameters)

; --- exports: explicitly public/open top-level declarations ----------------

(source_file (class_declaration) @reference.export)
(source_file (protocol_declaration) @reference.export)
(source_file (function_declaration) @reference.export)
(source_file (property_declaration) @reference.export)

; --- identifier fallback (bindings excluded in _identifier_record) -----------

(simple_identifier) @reference.identifier
(type_identifier) @reference.identifier
