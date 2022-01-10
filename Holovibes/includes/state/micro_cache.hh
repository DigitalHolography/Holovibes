#pragma once

/*
 * ------- HOW TO CREATE MICRO CACHES ---------
 *
 * Micro caches needs two things from you: You declarating them and putting them
 * in the Global State Holder(GSH). We will go through these 2 things now:
 *
 * 1) Declaring a new MicroCache class: Use the macro NEW_MICRO_CACHE as follows:
 *    ```
 *    NEW_MICRO_CACHE(MicroCacheClassName,
 *         (type_var_1, name_var_1),
 *         (type_var_2, name_var_2),
 *         (type_var_3, name_var_3),
 *         ....
 *    )
 *    ```
 *    Example:
 *    ```
 *    NEW_MICRO_CACHE(ExampleCache,
 *         (int, a),
 *         (std::string, mdr),
 *         (long long, r42)
 *    )
 *	  ```
 * 
 * 	  If you want to have a micro-cache with initialized values, use the macro NEW_INITIAIZED_MICRO_CACHE as follows:
 *	  ```
 *    NEW_INITIALIZED_MICRO_CACHE(MicroCacheClassName,
 *         (type_var_1, name_var_1, value_var_1),
 *         (type_var_2, name_var_2, value_var_2),
 *         (type_var_3, name_var_3, value_var_3),
 *         ....
 *    )
 *    ```
 *    Example:
 *    ```
 *    NEW_INITIALIZED_MICRO_CACHE(ExampleCache,
 *         (int, a, 0),
 *         (std::string, mdr, "mdr"),
 *         (long long, r42, 42)
 *    )
 *	  ```
 * 2) Put it in the Global State Holder(GSH) class and set it to be the reference cache:
 *    What's a reference cache ? The cache which is considered to be the reference and the one
 *    other caches of the same type will refer to when you need to synchronize them.
 *    To do that, you just need to instantiate an object with the correct "Ref" Struct:
 *    ```
 *    ExampleCache::Ref my_cache;
 *    ```
 *    Otherwise (which means, everywhere but not in the GSH), you can use the "Cache" Struct:
 *    ```
 *    ExampleCache::Cache my_cache;
 *    ```
 *
 * ------- HOW TO USE MICRO CACHES -------
 *
 * Caches and reference caches are the same object but they have no method in common:
 *
 * 1) Caches:
 *    * Getter: Each object $X you declared has its own getter named get_$X which returns a const ref
 *    * Syncronize: the method synchronize() copy changed elements of the reference cache to this cache
 *
 * 2) Reference caches (Each one of these methods are only available from inside the GSH):
 *    * Trigger: Each object $X you declared has its own trigger named trigger_$X which will trigger on all
 *      other micro caches of the same type the need to update this variable at the next synchronize
 *    * Getter: Each object $X you declared has its own getter named get_$X_ref which returns a ref
 *    * Setter: Each object $X you declared has its own trigger named set_$X which set the variable
 *      and run the trigger_$X method
 *
 * ------- LIMITATIONS --------
 *
 * 1) If you have commas in your variable type, consider using the 'using' directive on it.
 * 2) You cant have a variable with the same type and the same variable name in the same file
 * 3) The macro can generate a lot of code.
 * 4) Some macro hacks were needed to pull this off,
 *    for now the macro works fine on any last gen compiler.
 */

namespace holovibes
{

/*! \brief Superclass for all microcaches */
struct MicroCache;

/*!
 * \brief Concept used to have better template selection over micro cache classes.
 *        Each class created with NEW_MICRO_CACHE respect this concept.
 */
template <class T>
concept MicroCacheDerived = std::is_base_of<MicroCache, T>::value;

struct MicroCache
{
  protected:
    /*! \brief static templated variable containing the reference cache for this type */
    template <MicroCacheDerived T>
    static inline T* cache_reference = nullptr;

    /*! \brief static templated variable contaning all the micro caches refering to the same reference */
    template <MicroCacheDerived T>
    static inline std::set<T*> micro_caches;
};
} // namespace holovibes

#include "micro_cache.hxx"
