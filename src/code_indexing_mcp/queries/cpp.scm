(class_specifier
  name: (type_identifier) @name) @definition.class

(field_declaration_list
  (function_definition
    declarator: (function_declarator
      declarator: (field_identifier) @name))) @definition.method

(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name)) @definition.function
