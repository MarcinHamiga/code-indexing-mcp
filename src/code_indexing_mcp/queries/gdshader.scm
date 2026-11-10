(function_definition
  (identifier) @name) @definition.function

(struct_definition
  name: (identifier) @name) @definition.struct

; A uniform is the shader's exposed parameter -- Godot surfaces it as a
; material property in the inspector -- so it indexes as one.
(uniform_declaration
  (identifier) @name) @definition.property

;; `const float A = 1.0, B = 2.0;` holds several init_declarators, and one node
;; can carry only one definition, so each declarator is its own definition. The
;; name is the declarator's own identifier, never one inside its value
;; (`const float C = A;` defines C). The extractor widens a lone declarator back
;; to its declaration.
(const_declaration
  (init_declarator
    declarator: (identifier) @name) @definition.constant)
