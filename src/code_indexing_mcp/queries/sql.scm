(create_table
  (object_reference
    name: (identifier) @name)) @definition.table

(create_view
  (object_reference
    name: (identifier) @name)) @definition.view

(create_materialized_view
  (object_reference
    name: (identifier) @name)) @definition.view

(create_index
  (identifier) @name) @definition.index

(create_function
  (object_reference
    name: (identifier) @name)) @definition.function

(create_trigger
  (keyword_trigger)
  .
  (object_reference
    name: (identifier) @name)) @definition.trigger

(create_type
  (object_reference
    name: (identifier) @name)) @definition.type
