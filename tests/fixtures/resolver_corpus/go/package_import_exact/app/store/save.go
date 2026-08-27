package store

// Save is the only project-wide declaration of its name, so the qualified
// import from main.go can prove the binding exactly.
func Save(items string) error {
	return nil
}
