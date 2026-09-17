import Foundation

class Greeter {
    let name: String

    init(name: String) {
        self.name = name
    }

    func greet() -> String {
        return "hi " + name
    }
}

func topLevel(first: Int, second: Int) -> Int {
    return first + second
}

struct Point {
    var x: Int
    var y: Int
}

protocol Runnable {
    func run()
}
