using System;
using System.Collections.Generic;
using Widget = Acme.Widgets.Gadget;
using static Acme.Util.Errors;

namespace Demo.Catalog;

public delegate int Transform(int value);

public enum State
{
    On,
    Off,
}

public interface IService
{
    void Run();

    string Name { get; }
}

public record User(string Name);

public struct Point : IMeasure
{
    public Point(int x) => X = x;

    public int X { get; }
}

public class Outer : IService
{
    private Dictionary<string, Gadget> cache;
    private int count = 0;

    public Outer()
    {
        this.count = 0;
    }

    public string Name => "outer";

    [Obsolete]
    public void Run()
    {
        void Local() { }

        Local();
        Gadget widget = new Gadget("key");
        this.count = widget.Size();
        widget.Ready = true;
        foreach (Gadget item in widget.Items())
        {
            Console.WriteLine(item.Name());
        }
        try
        {
            Fail("busy");
        }
        catch (InvalidOperationException error)
        {
            Log(error);
        }
        Func<int> measure = () => widget.Size();
        Console.WriteLine(measure());
    }

    public class Inner
    {
        public void Work() { }
    }
}
