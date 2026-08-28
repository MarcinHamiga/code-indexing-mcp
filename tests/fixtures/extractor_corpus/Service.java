package com.example.service;

import com.example.util.Items;
import com.example.util.*;
import static com.example.util.Errors.fail;

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
    private Map<String, List<Item>> cache;

    public Service() {
        this.value = LIMIT;
    }

    private int value;

    @Override
    public String name() {
        return "service";
    }

    public void run(Handler handler) throws IOException {
        Item marker = new Item(handler.items());
        this.value = marker.size();
        handler.done = true;
        for (Item item : handler.items()) {
            System.out.println(item.name());
        }
        try {
            fail("busy");
        } catch (IllegalStateException error) {
            error.printStackTrace();
        }
        Runnable task = () -> handler.flush();
        Items.join(task);
    }

    static class Helper extends Service {
        int scaled(int input) {
            return input * LIMIT;
        }
    }
}
