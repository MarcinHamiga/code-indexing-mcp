; Sections are the only named structure in a scene or resource file, and what
; names them differs by section: a scene node carries `name`, while a resource
; reference carries `id`. Each pattern is anchored to its own section heading so
; a section carrying both cannot match twice -- matches arrive in source order
; rather than pattern order, so which one won would otherwise be arbitrary.
;
; `[gd_scene]`, `[gd_resource]`, `[resource]`, `[connection]`, and the sections
; of a `project.godot` name nothing, and reach the index as module text.

(section
  (identifier) @_section
  (attribute
    (identifier) @_key
    (string) @name)
  (#any-of? @_section "sub_resource" "ext_resource")
  (#eq? @_key "id")) @definition.object

(section
  (identifier) @_section
  (attribute
    (identifier) @_key
    (string) @name)
  (#eq? @_section "node")
  (#eq? @_key "name")) @definition.object
