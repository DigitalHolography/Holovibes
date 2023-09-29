# View Cache

### Window Kind Enum

```cpp
WindowKind {  
    ViewXY = 0,   /* Main view */  
    ViewXZ,       /* view slice */  
    ViewYZ,       /* YZ view slice */  
    ViewFilter2D, /* ViewFilter2D view */  
}
```

### View Struct Diagram

Note : All the structs also override the operators: != (inequality) and << (stream insertion), they also provide a call to the macro SERIALIZE_JSON_STRUCT for serialization

```mermaid
classDiagram
    class ViewContrast{
        +bool enabled
        +bool auto_refresh = true
        +bool invert = false
        +float min = 1.f
        +float max = 1.f
    }

    ViewWindow --> "1" ViewContrast
    class ViewWindow{
        +bool log_enabled
        +ViewContrast contrast;
    }

    ViewXYZ--|>ViewWindow
    class ViewXYZ{
        +bool horizontal_flip
        +float rotation
        +uint output_image_accumulation

        +is_image_accumulation_enabled() bool
    }

    Window --> "3" ViewXYZ
    Window --> "1" ViewWindow
    class Window{
        +ViewXYZ xy
        +ViewXYZ yz
        +ViewXYZ xz
        +ViewWindow filter2d

        +Update()
        +Load()
    }

    namespace AccumulationView{
        class ViewAccu{
            +int width
        }

        class ViewAccuPQ{
            +unsigned start
        }

        class ViewAccuXY{
            +unsigned start
        }
    }
    ViewAccuPQ--|>ViewAccu
    ViewAccuXY--|>ViewAccu

    class ReticleStruct{
        +bool display_enabled
        +float scale
    }

    Views --> "1" Window
    Views --> "2" ViewAccuXY
    Views --> "2" ViewAccuPQ
    Views --> "1" ReticleStruct
    class Views{
        +ImageTypeEnum image_type
        +bool fft_shift
        +ViewAccuXY x
        +ViewAccuXY y
        +ViewAccuPQ z
        +ViewAccuPQ z2
        +Windows window
        +bool renorm
        +ReticleStruct reticle

        +Update()
        +Load()
    }
```