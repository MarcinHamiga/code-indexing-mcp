package sample

import kotlin.math.max
import sample.store.*

class Greeter(val name: String) {
    fun greet(): String {
        return "hi " + name
    }
}

fun topLevel(first: Int, second: Int = 1): Int = max(first, second)

object Singletons {
    const val VERSION = 1
}

enum class Direction {
    NORTH,
    SOUTH
}

typealias Name = String
