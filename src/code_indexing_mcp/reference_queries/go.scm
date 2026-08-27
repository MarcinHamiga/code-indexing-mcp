; Structural reference captures for Go. Roots marked @reference.* are forwarded
; to _go_records, which re-dispatches on node.type; sub-captures are decorative.

; --- imports: one row per spec (plain, aliased, grouped, dot) -------------

(import_spec path: (interpreted_string_literal) @module) @reference.import

; --- calls: plain identifier calls and selector calls ----------------------

(call_expression function: (_) @name arguments: (_) @arguments) @reference.call

; --- non-call member access (reads and writes through selectors) -----------

(selector_expression) @reference.member_access

; --- type-bearing declarations (handler descends the type field) ----------

(field_declaration) @reference.field_declaration
(type_spec) @reference.type_spec
(type_alias) @reference.type_alias
(var_spec) @reference.var_spec
(parameter_declaration type: (_)) @reference.parameter_type
(variadic_parameter_declaration) @reference.variadic_parameter
(composite_literal type: (_)) @reference.composite_type
(qualified_type) @reference.qualified_type

; --- exports: top-level declarations only (qualification in the handler) --
; The capture must sit on the declaration node itself; annotating the outer
; parens would hand the handler the whole source_file.

(source_file (function_declaration) @reference.export)
(source_file (method_declaration) @reference.export)
(source_file (type_declaration) @reference.export)
(source_file (var_declaration) @reference.export)
(source_file (const_declaration) @reference.export)

; --- declaration parameters (receiver first -- slot 0 mirrors Python self) -

(function_declaration parameters: (parameter_list) @declaration.parameters)

(method_declaration
  receiver: (parameter_list) @declaration.parameters
  parameters: (parameter_list) @declaration.parameters)

; --- identifier fallback (bindings excluded in _identifier_record) --------

(identifier) @reference.identifier
(type_identifier) @reference.identifier
(field_identifier) @reference.identifier
