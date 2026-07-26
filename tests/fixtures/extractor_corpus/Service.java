package com.example.service;

public interface Marker {
    String name();
}

public enum Status {
    ACTIVE,
    CLOSED
}

public record Point(int x, int y) {}

public class Service implements Marker {
    private static final int LIMIT = 10;

    public Service() {
        this.value = LIMIT;
    }

    private int value;

    @Override
    public String name() {
        return "service";
    }

    static class Helper {
        int scaled(int input) {
            return input * LIMIT;
        }
    }
}
