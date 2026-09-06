; Structural reference captures for C. Roots marked @reference.* are forwarded
; to _c_records, which re-dispatches on node.type; sub-captures are decorative.

; --- includes: one row per directive (quoted and system) -------------------

(preproc_include path: (_) @module) @reference.import

; --- calls: plain identifier calls (member calls own their field_expression)

(call_expression function: (_) @name arguments: (_) @arguments) @reference.call

; --- non-call member access through `.` and `->` ----------------------------

(field_expression) @reference.member_access

; --- type-bearing declarations (the handler descends the type field) --------

(declaration) @reference.declaration

(field_declaration) @reference.declaration

(parameter_declaration) @reference.declaration

(type_definition) @reference.declaration

; --- exports and return types: top-level non-`static` linkage only ----------

(function_definition) @reference.export

; --- declaration parameters -------------------------------------------------

(function_definition
  declarator: (function_declarator
    parameters: (parameter_list) @declaration.parameters))

; --- identifier fallback (bindings excluded in _identifier_record) -----------

(identifier) @reference.identifier

(type_identifier) @reference.identifier

(field_identifier) @reference.identifier
