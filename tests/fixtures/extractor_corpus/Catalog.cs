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

public struct Point
{
    public Point(int x) => X = x;

    public int X { get; }
}

public class Outer : IService
{
    public Outer() { }

    public string Name => "outer";

    public void Run()
    {
        void Local() { }

        Local();
    }

    public class Inner
    {
        public void Work() { }
    }
}
