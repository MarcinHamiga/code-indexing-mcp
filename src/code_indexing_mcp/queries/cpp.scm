; Only a specifier with a body declares the type; `class X;`, `struct X *p;`
; and `Widget w;` mention it.
(class_specifier
  name: (type_identifier) @name
  body: (field_declaration_list)) @definition.class

(struct_specifier
  name: (type_identifier) @name
  body: (field_declaration_list)) @definition.struct

; Inline methods and free functions share one shape: the definition is the
; function_definition itself, never the enclosing field_declaration_list. The
; extractor turns a function inside a class or struct into a method.
(function_definition
  declarator: (function_declarator
    declarator: [
      (identifier)
      (field_identifier)
    ] @name)) @definition.function

; Out-of-class definitions: `void Stack::push() {}`.
(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier
      name: (identifier) @name))) @definition.function

; Pointer and reference returns (`Widget* make()`, `Widget& instance()`),
; including `Widget* Stack::build()`.
(function_definition
  declarator: [
    (pointer_declarator
      declarator: (function_declarator
        declarator: [
          (identifier)
          (field_identifier)
        ] @name))
    (reference_declarator
      (function_declarator
        declarator: [
          (identifier)
          (field_identifier)
        ] @name))
    (pointer_declarator
      declarator: (function_declarator
        declarator: (qualified_identifier
          name: (identifier) @name)))
    (reference_declarator
      (function_declarator
        declarator: (qualified_identifier
          name: (identifier) @name)))
  ]) @definition.function
