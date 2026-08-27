package main

import "app/store"

// The enclosing method's receiver parameter is `w`, not `s`, so the receiver
// name cannot prove which object `s` holds -- even though `Handle` is unique.
func (w *Worker) Run() {
	s := store.NewStore()
	s.Handle()
}
