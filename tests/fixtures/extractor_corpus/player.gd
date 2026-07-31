class_name Player
extends CharacterBody2D

signal health_changed(amount: int)

enum State { IDLE, RUN }

const MAX_SPEED := 300.0

var health: int = 100

class Inventory:
	var items := []

	func add(item) -> void:
		items.append(item)

static func make() -> Player:
	return Player.new()

func _ready() -> void:
	health_changed.emit(health)
