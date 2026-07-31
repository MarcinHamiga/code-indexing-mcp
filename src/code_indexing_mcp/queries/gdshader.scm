(function_definition
  (identifier) @name) @definition.function

(struct_definition
  name: (identifier) @name) @definition.struct

; A uniform is the shader's exposed parameter -- Godot surfaces it as a
; material property in the inspector -- so it indexes as one.
(uniform_declaration
  (identifier) @name) @definition.property

(const_declaration
  (init_declarator
    (identifier) @name)) @definition.constant
