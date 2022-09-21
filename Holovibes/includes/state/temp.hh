NEW_MICRO_CACHE(ExampleCache, (int, example))

struct ExampleCache
{
    ExampleCache::Ref
    {
        set_example(int value);
        get_example();
    };

    ExampleCache::Cache
    {
        void synchronize();
        get_example();
    };
};
