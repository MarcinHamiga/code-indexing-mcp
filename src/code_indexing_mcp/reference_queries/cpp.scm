; Structural reference captures for C++. The shared C handler (_c_records)
; owns includes, calls, member access, declarations, and linkage exports;
; _cpp_records adds inheritance, alias values, constructor calls, and type
; exports below. Sub-captures are decorative.

; --- includes: one row per directive (quoted and system) -------------------

(preproc_include path: (_) @module) @reference.import

; --- calls: plain, qualified (`A::create`), and member calls ---------------

(call_expression function: (_) @name arguments: (_) @arguments) @reference.call

; --- constructor calls (`new Widget(1)`) ------------------------------------

(new_expression) @reference.constructor

; --- non-call member access through `.` and `->` ----------------------------

(field_expression) @reference.member_access

; --- base clauses (the handler emits inheritance per base type) --------------

(base_class_clause) @reference.inheritance

; --- type-bearing declarations (the handler descends the type field) --------

(declaration) @reference.declaration

(field_declaration) @reference.declaration

(parameter_declaration) @reference.declaration

(optional_parameter_declaration) @reference.declaration

(type_definition) @reference.declaration

(alias_declaration) @reference.declaration

; --- exports: top-level linkage and type declarations only ------------------

(function_definition) @reference.export

(class_specifier) @reference.export

(struct_specifier) @reference.export

; --- declaration parameters -------------------------------------------------

(function_definition
  declarator: (function_declarator
    parameters: (parameter_list) @declaration.parameters))

; --- identifier fallback (bindings excluded in _identifier_record) -----------

(identifier) @reference.identifier

(type_identifier) @reference.identifier

(field_identifier) @reference.identifier

(namespace_identifier) @reference.identifier
