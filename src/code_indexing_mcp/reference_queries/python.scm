(import_from_statement
  module_name: (dotted_name) @module
  name: (aliased_import
    name: (dotted_name) @name
    alias: (identifier) @alias)) @reference.import

(import_from_statement
  module_name: (dotted_name) @module
  name: (dotted_name) @name) @reference.import

(import_statement
  name: (aliased_import
    name: (dotted_name) @name
    alias: (identifier) @alias)) @reference.import

(import_statement name: (dotted_name) @name) @reference.import

(decorator
  (call function: (_) @name arguments: (argument_list) @arguments)) @reference.decorator

(decorator (_) @name) @reference.decorator

(class_definition
  superclasses: (argument_list (_) @name)) @reference.inheritance

(call function: (_) @name arguments: (argument_list) @arguments) @reference.call

(call
  function: (attribute object: (_) @receiver attribute: (_) @name)
  arguments: (argument_list) @arguments) @reference.call

(type (_) @name) @reference.type_use

(function_definition
  name: (identifier) @name
  parameters: (parameters) @declaration.parameters)

(class_definition name: (identifier) @name)
