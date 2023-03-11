
class Cache
{
public:
	Cache();
	~Cache();

	int write_cache(int frame);
	int read_cache(int frame);
	int check_cache(int frame);
	int cache_free(int frame);

private:

};

Cache::Cache()
{
}

Cache::~Cache()
{
}