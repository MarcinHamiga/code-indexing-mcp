; Structural reference captures for C#. Roots marked @reference.* are
; forwarded to _csharp_records, which re-dispatches on node.type; sub-captures
; are decorative.

; --- imports: one row per directive (namespace, static, alias, global) ------

(using_directive) @reference.import

; --- calls: invocations and constructor object creation ---------------------

(invocation_expression) @reference.call
(object_creation_expression) @reference.call

; --- non-call member access (reads and writes through member access) --------

(member_access_expression) @reference.member_access

; --- heritage, attributes, constraints, and casts ----------------------------

(base_list) @reference.inheritance
(attribute) @reference.decorator
(cast_expression) @reference.cast_type
(type_parameter_constraints_clause) @reference.constraints

; --- type expressions (handler descends to the naming leaves) ---------------

(generic_name) @reference.generic
(qualified_name) @reference.qualified
(array_type) @reference.array
(nullable_type) @reference.nullable

; --- declaration type fields (handler descends the type position) ----------

(method_declaration) @reference.decl_type
(property_declaration) @reference.decl_type
(variable_declaration) @reference.decl_type
(parameter) @reference.decl_type
(catch_declaration) @reference.decl_type
(foreach_statement) @reference.decl_type

; --- exports: top-level types, any accessibility ----------------------------
; The namespace module path is attached in the handler by climbing to the
; enclosing namespace declaration (D2).

(class_declaration) @reference.export
(struct_declaration) @reference.export
(interface_declaration) @reference.export
(record_declaration) @reference.export
(enum_declaration) @reference.export
(delegate_declaration) @reference.export

; --- declaration parameters -------------------------------------------------

(method_declaration parameters: (parameter_list) @declaration.parameters)

(constructor_declaration
  parameters: (parameter_list) @declaration.parameters)

; --- identifier fallback (bindings excluded in _identifier_record) ----------

(identifier) @reference.identifier
