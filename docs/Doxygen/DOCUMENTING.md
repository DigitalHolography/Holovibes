## Documenting using Doxygen

### General format

For multi-line comments, use:

```{.cc}
/*! 
 *
 */
```

For single line comments, prefer using:
```{.cc}
/*!          */
```

### Files

Always give at least a brief description and a `\file` tag for every file. This is especially important for .cuh files as they might not be recognized in the resulting documentation otherwise.

```{.cc}
/*! \file
 *
 * \brief Description
 *
 * (Optional)
 * Detailed description
 * Another line of detailed description
 */
```

### Classes and Enums

Always give at least a brief description for every class or enum

```{.cc}
/*! \class MyClass OR \enum MyEnum
 *
 * \brief Description
 *
 * (Optional)
 * Detailed description
 * Another line of detailed description
 */
```

#### Enum members

If you comment a member of a given enum, comment every other member of said enum.

Use either:

```{.cc}
/*! \brief Member description */
Member,
```

Or, if you prefer inlining:
```{.cc}
Member, /*!< Member description*/
```

But not both in the same enum.

### Functions and methods

When commenting a function or method, use:

```{.cc}
/*! brief Description
 *
 * (Optional)
 * Detailed description
 * Another line of detailed description
 *
 * (Optional)
 * /param test Test variable
 * /return What is being returned
 *
```

### Variables and attributes

When commenting a variable or an attribute, use:

```{.cc}
/*! \brief Description
 *
 * (Optional)
 * Detailed description
 * Another line of detailed description
 */
```

### Grouping methods or attributes

When grouping methods or attributes in the same category, use:

```{.cc}
/*! \name Category Name
 * \{
 */
/*! <Documentation for attribute> */
int attribute = 1;
/*! <Documentation for method> */
void someMethod();
/*! \} */
```