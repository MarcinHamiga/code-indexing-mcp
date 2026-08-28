; Structural reference captures for Java. Roots marked @reference.* are
; forwarded to _java_records, which re-dispatches on node.type; sub-captures
; are decorative.

; --- imports: one row per declaration (single-type, static, on-demand) -----

(import_declaration) @reference.import

; --- calls: method invocations and constructor object creation -------------

(method_invocation) @reference.call
(object_creation_expression) @reference.call

; --- non-call member access (reads and writes through field access) --------

(field_access) @reference.member_access

; --- heritage, throws, bounds, and catch types -----------------------------

(superclass) @reference.superclass
(super_interfaces) @reference.super_interfaces
(extends_interfaces) @reference.extends_interfaces
(throws) @reference.throws
(type_bound) @reference.type_bound
(catch_type) @reference.catch_type

; --- type expressions (handler descends to the naming leaves) --------------

(generic_type) @reference.generic_type
(scoped_type_identifier) @reference.scoped_type
(array_type) @reference.array_type

; --- annotations (decorator rows) -------------------------------------------

(marker_annotation) @reference.decorator
(annotation) @reference.decorator

; --- declaration type fields (handler descends the `type` field) -----------

(method_declaration) @reference.decl_type
(field_declaration) @reference.decl_type
(local_variable_declaration) @reference.decl_type
(constant_declaration) @reference.decl_type
(formal_parameter) @reference.decl_type
(spread_parameter) @reference.decl_type
(enhanced_for_statement) @reference.decl_type

; --- exports: public top-level types only ----------------------------------

(program (class_declaration) @reference.export)
(program (interface_declaration) @reference.export)
(program (record_declaration) @reference.export)
(program (enum_declaration) @reference.export)
(program (annotation_type_declaration) @reference.export)

; --- declaration parameters -------------------------------------------------

(method_declaration parameters: (formal_parameters) @declaration.parameters)

(constructor_declaration
  parameters: (formal_parameters) @declaration.parameters)

(record_declaration
  parameters: (formal_parameters) @declaration.parameters)

; --- identifier fallback (bindings excluded in _identifier_record) ----------

(identifier) @reference.identifier
(type_identifier) @reference.identifier
