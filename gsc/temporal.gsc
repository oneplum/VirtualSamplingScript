setdir output
set fileformat png
setinteraction 0
logfile temporal.log
clearlog
logGLInfo true
resize 512 512

repeat 9 as volume
    setLevel 0
    settranslation 0 0 0
    resetrotation
    if $volume == 0
        #http://www.cgvis.de/vs/?dataset=6&method=7&const=0&ortho=1&vs=6&sr=0&tfType=0&tfStart=0.097&tfWidth=0.156&alpha=0.9900000095367432&level=0&transform=-0.999435186_-0.0331454054_-0.00540661532_0_0.00595936924_-0.0165996496_-0.999840856_0_0.0330504514_-0.999313653_0.0167886354_0_0_0_0_1%7E0_0_0
        set volname Head1
        set analytic false
        setvolume 6
        settfparams false 0.097 0.156
    endif
    if $volume == 1
        # http://www.cgvis.de/vs/?dataset=6&method=13&const=0&ortho=1&vs=6&sr=0&tfType=0&tfStart=0.218&tfWidth=0.142&alpha=0.9900000095367432&level=0&transform=-0.136617959_0.990499735_-0.015709335_0_0.0695978329_-0.00622117426_-0.997553885_0_-0.988174617_-0.137377352_-0.0680867285_0_0_0_0_1%7E0_0_0
        set volname Head2
        set analytic false
        setvolume 6
        settfparams false 0.218 0.142
    endif
    if $volume == 2
        #http://www.cgvis.de/vs/?dataset=7&method=0&const=0&ortho=0&vs=10&sr=3&tfType=2&tfStart=0.02&tfWidth=0.003&alpha=0.9900000095367432&level=0&transform=-0.202214763_0.978688002_-0.0357658304_0_-0.202166244_-0.00598103832_0.979332626_0_0.958247483_0.205267921_0.199067056_0_0_0_0_1%7E0_0_1.50000024&tfEncoding=UkxFRgAEAAAAAAAAAgABAQEKAQ8BEgEYARoBHQEjASYBLgExATQBNgI5AT8BQQJEAkcDSgFNAU8CUgNVAVsBXQJjAWYCaQFrAW4CdAF2AXkBfAF_AYQBhwGKAY0BkAGSApUCmwGdAaABowGmAakBqwGuAbEBtAG3AbkBvAG_AcIBxwHKAc0B0gHVAdgB3gLgA-MB6wLpAesB7gHxAvQC9wH0Afeg_wErATEBPwFdAWYBbgF2AX8BhwGNAZICmAGbAZ0BoAGjAaYBqQGrAa4DsQG0AbcBuQG8Ar8BwgLEAccBygLNAdAB0gHVAtgB3gHbAd4B4AHjAukB6wHuAfEB9AH3AfzJ_wIAAQ0KAAEEAgABAQIHAQoEAAEBAQcBCgIPARIBFQEaAR0BIAEmASgBKwEuAjQBOQE_AUEBRAFHAUoCTQJSAVUBWAFbAVIBWwFdAWABYwFmAWkBawJxAXYBfAF_A4IBhAGHAooBkAGSAZgBmwGdAZUBpgGrAa4BpgG0AbcBuQK8Ar8BwgLEAcoCzQLQAdIB1QLYAdsB3gHgAeMB5gHpAesB7gH0AfkE_wHxAfcB_In_AQADBAIPARIBGAEaAR0BIAEjASgBLgE0ATkBPAFBAUcCSgFNAU8BVQFYAV0CZgFpAW4BcQKCAYoBkAGSAZUBmAKdAaMBqQGrArEBtAG3AbkBvAG_AcQCxwHNAdAB0gHVAdgB2wHeAuAB4wHpAu4B8QH0AfkB_Lr_
        set volname Tree1
        set analytic false
        setvolume 7
        settfcode UkxFRgAEAAAAAAAAAgABAQEKAQ8BEgEYARoBHQEjASYBLgExATQBNgI5AT8BQQJEAkcDSgFNAU8CUgNVAVsBXQJjAWYCaQFrAW4CdAF2AXkBfAF_AYQBhwGKAY0BkAGSApUCmwGdAaABowGmAakBqwGuAbEBtAG3AbkBvAG_AcIBxwHKAc0B0gHVAdgB3gLgA-MB6wLpAesB7gHxAvQC9wH0Afeg_wErATEBPwFdAWYBbgF2AX8BhwGNAZICmAGbAZ0BoAGjAaYBqQGrAa4DsQG0AbcBuQG8Ar8BwgLEAccBygLNAdAB0gHVAtgB3gHbAd4B4AHjAukB6wHuAfEB9AH3AfzJ_wIAAQ0KAAEEAgABAQIHAQoEAAEBAQcBCgIPARIBFQEaAR0BIAEmASgBKwEuAjQBOQE_AUEBRAFHAUoCTQJSAVUBWAFbAVIBWwFdAWABYwFmAWkBawJxAXYBfAF_A4IBhAGHAooBkAGSAZgBmwGdAZUBpgGrAa4BpgG0AbcBuQK8Ar8BwgLEAcoCzQLQAdIB1QLYAdsB3gHgAeMB5gHpAesB7gH0AfkE_wHxAfcB_In_AQADBAIPARIBGAEaAR0BIAEjASgBLgE0ATkBPAFBAUcCSgFNAU8BVQFYAV0CZgFpAW4BcQKCAYoBkAGSAZUBmAKdAaMBqQGrArEBtAG3AbkBvAG_AcQCxwHNAdAB0gHVAdgB2wHeAuAB4wHpAu4B8QH0AfkB_Lr_
    endif
    if $volume == 3
        # http://www.cgvis.de/vs/?dataset=7&method=7&const=0&ortho=0&vs=10&sr=0&tfType=2&tfStart=0.02&tfWidth=0.003&alpha=0.9900000095367432&level=0&transform=-0.969087362_0.246711731_0.00148320443_0_-0.012833355_-0.0564099029_0.998324871_0_0.246383488_0.96744591_0.0578331985_0_0_0_0_1%7E0_0_0.800000072&tfEncoding=UkxFRgAEAAAAAAAAARUDGAQaAx0CIAUjAiYBKAErAi4BMQE0ATYBOQE8AkEBRAFHAkoBTQJPAVIBVQJYAVsBXQFgAWMCZgJpAWsBbgFxAnQBdgF5AXwBggGEAYoBjQGQAZIBlQKYAZsBnQKgAaMBpgGpAasBuQK_AcQBygHNAdAB0gHVAdgB2wLeAeAB4wHpAesB7gHxAfQB9___of8CUgRPB00FTwJSBFUGWANbAl0CYAFjAmYCaQJrA3ENdAJ2AnkDfAJ_AoIBhAGHBIoCjQGQApIBlQKYAZsCnQGgAaMCpgKrAbEBtAG3AbkBvALCAccBzQHQAdUB2AHbAd4C4AHjAeYB6QLrAe4B8QH3AfyJ_wQA_P8
        set volname Tree2
        set analytic false
        setvolume 7
        settfcode UkxFRgAEAAAAAAAAAgABAQEKAQ8BEgEYARoBHQEjASYBLgExATQBNgI5AT8BQQJEAkcDSgFNAU8CUgNVAVsBXQJjAWYCaQFrAW4CdAF2AXkBfAF_AYQBhwGKAY0BkAGSApUCmwGdAaABowGmAakBqwGuAbEBtAG3AbkBvAG_AcIBxwHKAc0B0gHVAdgB3gLgA-MB6wLpAesB7gHxAvQC9wH0Afeg_wErATEBPwFdAWYBbgF2AX8BhwGNAZICmAGbAZ0BoAGjAaYBqQGrAa4DsQG0AbcBuQG8Ar8BwgLEAccBygLNAdAB0gHVAtgB3gHbAd4B4AHjAukB6wHuAfEB9AH3AfzJ_wIAAQ0KAAEEAgABAQIHAQoEAAEBAQcBCgIPARIBFQEaAR0BIAEmASgBKwEuAjQBOQE_AUEBRAFHAUoCTQJSAVUBWAFbAVIBWwFdAWABYwFmAWkBawJxAXYBfAF_A4IBhAGHAooBkAGSAZgBmwGdAZUBpgGrAa4BpgG0AbcBuQK8Ar8BwgLEAcoCzQLQAdIB1QLYAdsB3gHgAeMB5gHpAesB7gH0AfkE_wHxAfcB_In_AQADBAIPARIBGAEaAR0BIAEjASgBLgE0ATkBPAFBAUcCSgFNAU8BVQFYAV0CZgFpAW4BcQKCAYoBkAGSAZUBmAKdAaMBqQGrArEBtAG3AbkBvAG_AcQCxwHNAdAB0gHVAdgB2wHeAuAB4wHpAu4B8QH0AfkB_Lr_
    endif
    if $volume == 4
        # http://www.cgvis.de/vs/?dataset=4&method=39&const=0&ortho=1&vs=6&sr=0&tfType=1&tfStart=0.075&tfWidth=0.01&alpha=0.9900000095367432&level=0&transform=0.803756475_-0.590385795_-0.0736123994_0_0.464429945_0.699930787_-0.542587698_0_0.371860117_0.401920915_0.836767256_0_0_0_0_1%7E0_0_0.700000048
        set volname Aneurism1
        set analytic false
        setvolume 4
        settfparams true 0.075 0.01
    endif
    if $volume == 5
        # http://www.cgvis.de/vs/?dataset=4&method=58&const=0&ortho=1&vs=4&sr=0&tfType=2&tfStart=0&tfWidth=0.528&alpha=0.9900000095367432&level=0&transform=0.803756475_-0.590385795_-0.0736123994_0_0.464429945_0.699930787_-0.542587698_0_0.371860117_0.401920915_0.836767256_0_0_0_0_1%7E0_0_0.700000048&tfEncoding=UkxFRgAEAAAAAAAAAgQBBQEGAQcBCAEJAQoBDAENAQ4BEAERARMBFAEWARgBGgEbAR0BHwEhASMBJQEnASkBLAEuATABMgE1ATcBOgE8AT4BQQFDAUYBSQFLAU4BUAFTAVYBWQFbAV4BYQFkAWYBaQFsAW8BcgF0AXcBegF9AYABgwGFAYgBiwGOAZEBlAGWAZkBnAGfAaEBpAGnAaoBrAGvAbIBtAG3AbkBvAG-AcEBwwHGAcgBywHNAc8B0QHUAdYB2AHaAdwB3gHgAeIB5AHmAecB6QHrAewB7gHvAfEB8gHzAfUB9gH3AfgB-QH6AfsC_AL9Av6G_wIEAQUBBgEHAQgBCQEKAQwBDQEOARABEQETARQBFgEYARoBGwEdAR8BIQEjASUBJwEpASwBLgEwATIBNQE3AToBPAE-AUEBQwFGAUkBSwFOAVABUwFWAVkBWwFeAWEBZAFmAWkBbAFvAXIBdAF3AXoBfQGAAYMBhQGIAYsBjgGRAZQBlgGZAZwBnwGhAaQBpwGqAawBrwGyAbQBtwG5AbwBvgHBAcMBxgHIAcsBzQHPAdEB1AHWAdgB2gHcAd4B4AHiAeQB5gHnAekB6wHsAe4B7wHxAfIB8wH1AfYB9wH4AfkB-gH7AvwC_QL-hv8CBAEFAQYBBwEIAQkBCgEMAQ0BDgEQAREBEwEUARYBGAEaARsBHQEfASEBIwElAScBKQEsAS4BMAEyATUBNwE6ATwBPgFBAUMBRgFJAUsBTgFQAVMBVgFZAVsBXgFhAWQBZgFpAWwBbwFyAXQBdwF6AX0BgAGDAYUBiAGLAY4BkQGUAZYBmQGcAZ8BoQGkAacBqgGsAa8BsgG0AbcBuQG8Ab4BwQHDAcYByAHLAc0BzwHRAdQB1gHYAdoB3AHeAeAB4gHkAeYB5wHpAesB7AHuAe8B8QHyAfMB9QH2AfcB-AH5AfoB-wL8Av0C_ob_BQABCAEJAQoBDAENAQ4BEAERARMBFAEWARgBGgEbAR0BHwEhASMBJQEnASkBLAEuATABMgE1ATcBOgE8AT4BQQFDAUYBSQFLAU4BUAFTAVYBWQFbAV4BYQFkAWYBaQFsAW8BcgF0AXcBegF9AYABgwGFAYgBiwGOAZEBlAGWAZkBnAGfAaEBpAGnAaoBrAGvAbIBtAG3AbkBvAG-AcEBwwHGAcgBywHNAc8B0QHUAdYB2AHaAdwB3gHgAeIB5AHmAecB6QHrAewB7gHvAfEB8gHzAfUB9gH3AfgB-QH6AfsC_AL9Av6G_w
        set volname Aneurism2
        set analytic false
        setvolume 4
        settfcode UkxFRgAEAAAAAAAAAgQBBQEGAQcBCAEJAQoBDAENAQ4BEAERARMBFAEWARgBGgEbAR0BHwEhASMBJQEnASkBLAEuATABMgE1ATcBOgE8AT4BQQFDAUYBSQFLAU4BUAFTAVYBWQFbAV4BYQFkAWYBaQFsAW8BcgF0AXcBegF9AYABgwGFAYgBiwGOAZEBlAGWAZkBnAGfAaEBpAGnAaoBrAGvAbIBtAG3AbkBvAG-AcEBwwHGAcgBywHNAc8B0QHUAdYB2AHaAdwB3gHgAeIB5AHmAecB6QHrAewB7gHvAfEB8gHzAfUB9gH3AfgB-QH6AfsC_AL9Av6G_wIEAQUBBgEHAQgBCQEKAQwBDQEOARABEQETARQBFgEYARoBGwEdAR8BIQEjASUBJwEpASwBLgEwATIBNQE3AToBPAE-AUEBQwFGAUkBSwFOAVABUwFWAVkBWwFeAWEBZAFmAWkBbAFvAXIBdAF3AXoBfQGAAYMBhQGIAYsBjgGRAZQBlgGZAZwBnwGhAaQBpwGqAawBrwGyAbQBtwG5AbwBvgHBAcMBxgHIAcsBzQHPAdEB1AHWAdgB2gHcAd4B4AHiAeQB5gHnAekB6wHsAe4B7wHxAfIB8wH1AfYB9wH4AfkB-gH7AvwC_QL-hv8CBAEFAQYBBwEIAQkBCgEMAQ0BDgEQAREBEwEUARYBGAEaARsBHQEfASEBIwElAScBKQEsAS4BMAEyATUBNwE6ATwBPgFBAUMBRgFJAUsBTgFQAVMBVgFZAVsBXgFhAWQBZgFpAWwBbwFyAXQBdwF6AX0BgAGDAYUBiAGLAY4BkQGUAZYBmQGcAZ8BoQGkAacBqgGsAa8BsgG0AbcBuQG8Ab4BwQHDAcYByAHLAc0BzwHRAdQB1gHYAdoB3AHeAeAB4gHkAeYB5wHpAesB7AHuAe8B8QHyAfMB9QH2AfcB-AH5AfoB-wL8Av0C_ob_BQABCAEJAQoBDAENAQ4BEAERARMBFAEWARgBGgEbAR0BHwEhASMBJQEnASkBLAEuATABMgE1ATcBOgE8AT4BQQFDAUYBSQFLAU4BUAFTAVYBWQFbAV4BYQFkAWYBaQFsAW8BcgF0AXcBegF9AYABgwGFAYgBiwGOAZEBlAGWAZkBnAGfAaEBpAGnAaoBrAGvAbIBtAG3AbkBvAG-AcEBwwHGAcgBywHNAc8B0QHUAdYB2AHaAdwB3gHgAeIB5AHmAecB6QHrAewB7gHvAfEB8gHzAfUB9gH3AfgB-QH6AfsC_AL9Av6G_w
    endif
    if $volume == 6
        # http://www.cgvis.de/vs/?dataset=0&method=79&const=0&ortho=0&vs=1&sr=0&tfType=2&tfStart=0.576&tfWidth=0&alpha=0.9900000095367432&level=0&transform=-0.94459784_-0.268845826_-0.18829757_0_-0.200813919_0.927143097_-0.316352993_0_0.259629518_-0.261014193_-0.929764926_0_0_0_0_1%7E0_0_0.100000016&tfEncoding=UkxFRgAEAAAAAAAANwAV__8AGgAL__8AOAAZ_30ACv8hAAMgAR0BEgEBCgABATYADf9EAA
        set volname Sphere
        set analytic true
        setvolume 0
        settfcode UkxFRgAEAAAAAAAANwAV__8AGgAL__8AOAAZ_30ACv8hAAMgAR0BEgEBCgABATYADf9EAA
        setbackground 1 1 1 1
    endif
    if $volume == 7
        # https://www.cgvis.de/vs/?dataset=2&method=52&const=0&ortho=0&vs=1&sr=10&tfType=1&tfStart=0.397&tfWidth=0.006&alpha=0.9900000095367432&level=0&transform=0.84378463_-0.535941601_0.0281707458_0_0.262970418_0.458637923_0.848821521_0_-0.467838824_-0.708814919_0.527927518_0_0_0_0_1%7E0_0_0.5
        set volname ML1
        set analytic true
        setvolume 2
        settfparams true 0.397 0.006
        setbackground 0 0 1 1
    endif
    if $volume == 8
        # http://www.cgvis.de/vs/?dataset=2&method=52&const=0&ortho=0&vs=1&sr=0&tfType=0&tfStart=0.38&tfWidth=0.09&alpha=0.9900000095367432&level=0&transform=0.84378463_-0.535941601_0.0281707458_0_0.262970418_0.458637923_0.848821521_0_-0.467838824_-0.708814919_0.527927518_0_0_0_0_1%7E0_0_0
        set volname ML2
        set analytic true
        setvolume 2
        settfparams false 0.38 0.09
        setbackground 0 0 1 1
    endif
    log Volume $volume
    repeat 2 as lighting
        if $lighting == 0
            log unlit pass
            set lit n
        else
            log lighting pass
            set lit l
            setLevel 0
            settranslation 0 0 0
            resetrotation
            if $volume == 0
                settfparams false 0.097 0.156
            endif
            if $volume == 1
                settfparams false 0.218 0.142
            endif
            if $volume == 2
                #http://www.cgvis.de/vs/?dataset=7&method=59&const=0&ortho=0&vs=10&sr=1&tfType=2&tfStart=0.02&tfWidth=0.003&alpha=0.9900000095367432&level=0&transform=-0.202214763_0.978688002_-0.0357658304_0_-0.202166244_-0.00598103832_0.979332626_0_0.958247483_0.205267921_0.199067056_0_0_0_0_1%7E0_0_1.50000024&tfEncoding=UkxFRgAEAAAAAAAAARUDGAQaAx0CIAUjAiYBKAErAi4BMQE0ATYBOQE8AkEBRAFHAkoBTQJPAVIBVQJYAVsBXQFgAWMCZgJpAWsBbgFxAnQBdgF5AXwBggGEAYoBjQGQAZIBlQKYAZsBnQKgAaMBpgGpAasBuQK_AcQBygHNAdAB0gHVAdgB2wLeAeAB4wHpAesB7gHxAfQB9___of8CUgRPB00FTwJSBFUGWANbAl0CYAFjAmYCaQJrA3ENdAJ2AnkDfAJ_AoIBhAGHBIoCjQGQApIBlQKYAZsCnQGgAaMCpgKrAbEBtAG3AbkBvALCAccBzQHQAdUB2AHbAd4C4AHjAeYB6QLrAe4B8QH3AfyJ_wQA_P8
                settfcode UkxFRgAEAAAAAAAAARUDGAQaAx0CIAUjAiYBKAErAi4BMQE0ATYBOQE8AkEBRAFHAkoBTQJPAVIBVQJYAVsBXQFgAWMCZgJpAWsBbgFxAnQBdgF5AXwBggGEAYoBjQGQAZIBlQKYAZsBnQKgAaMBpgGpAasBuQK_AcQBygHNAdAB0gHVAdgB2wLeAeAB4wHpAesB7gHxAfQB9___of8CUgRPB00FTwJSBFUGWANbAl0CYAFjAmYCaQJrA3ENdAJ2AnkDfAJ_AoIBhAGHBIoCjQGQApIBlQKYAZsCnQGgAaMCpgKrAbEBtAG3AbkBvALCAccBzQHQAdUB2AHbAd4C4AHjAeYB6QLrAe4B8QH3AfyJ_wQA_P8
            endif
            if $volume == 3
                # http://www.cgvis.de/vs/?dataset=7&method=7&const=0&ortho=0&vs=10&sr=0&tfType=2&tfStart=0.02&tfWidth=0.003&alpha=0.9900000095367432&level=0&transform=-0.969087362_0.246711731_0.00148320443_0_-0.012833355_-0.0564099029_0.998324871_0_0.246383488_0.96744591_0.0578331985_0_0_0_0_1%7E0_0_0.800000072&tfEncoding=UkxFRgAEAAAAAAAAARUDGAQaAx0CIAUjAiYBKAErAi4BMQE0ATYBOQE8AkEBRAFHAkoBTQJPAVIBVQJYAVsBXQFgAWMCZgJpAWsBbgFxAnQBdgF5AXwBggGEAYoBjQGQAZIBlQKYAZsBnQKgAaMBpgGpAasBuQK_AcQBygHNAdAB0gHVAdgB2wLeAeAB4wHpAesB7gHxAfQB9___of8CUgRPB00FTwJSBFUGWANbAl0CYAFjAmYCaQJrA3ENdAJ2AnkDfAJ_AoIBhAGHBIoCjQGQApIBlQKYAZsCnQGgAaMCpgKrAbEBtAG3AbkBvALCAccBzQHQAdUB2AHbAd4C4AHjAeYB6QLrAe4B8QH3AfyJ_wQA_P8
                settfcode UkxFRgAEAAAAAAAAARUDGAQaAx0CIAUjAiYBKAErAi4BMQE0ATYBOQE8AkEBRAFHAkoBTQJPAVIBVQJYAVsBXQFgAWMCZgJpAWsBbgFxAnQBdgF5AXwBggGEAYoBjQGQAZIBlQKYAZsBnQKgAaMBpgGpAasBuQK_AcQBygHNAdAB0gHVAdgB2wLeAeAB4wHpAesB7gHxAfQB9___of8CUgRPB00FTwJSBFUGWANbAl0CYAFjAmYCaQJrA3ENdAJ2AnkDfAJ_AoIBhAGHBIoCjQGQApIBlQKYAZsCnQGgAaMCpgKrAbEBtAG3AbkBvALCAccBzQHQAdUB2AHbAd4C4AHjAeYB6QLrAe4B8QH3AfyJ_wQA_P8
            endif
            if $volume == 4
                settfparams true 0.075 0.01
            endif
            if $volume == 5
                # http://www.cgvis.de/vs/?dataset=4&method=46&const=0&ortho=1&vs=1&sr=0&tfType=0&tfStart=0.213&tfWidth=0.005&alpha=0.9900000095367432&level=0&transform=0.803756475_-0.590385795_-0.0736123994_0_0.464429945_0.699930787_-0.542587698_0_0.371860117_0.401920915_0.836767256_0_0_0_0_1%7E0_0_0.700000048
                settfparams false 0.213 0.005
            endif
            if $volume == 6
                settfcode UkxFRgAEAAAAAAAANwAV__8AGgAL__8AOAAZ_30ACv8hAAMgAR0BEgEBCgABATYADf9EAA
            endif
            if $volume == 7
                settfparams true 0.397 0.006
            endif
            if $volume == 8
                settfparams false 0.397 0.006
            endif
        endif
        repeat 3 as l
            setLevel $l
            log Level $l
            repeat 6 as tmethod
                set m $tmethod * 13
                if $tmethod == 0
                    set tname lin
                endif
                if $tmethod == 1
                    set tname quadB
                endif
                if $tmethod == 2
                    set tname quadBf
                endif
                if $tmethod == 3
                    set tname cubicB
                endif
                if $tmethod == 4
                    set tname cubicBf
                endif
                if $tmethod == 5
                    set tname cr
                endif
                if $lighting == 1
                    set m $m + 7
                endif
                setmethod $m
                log Method $m
                repeat 5 as rateIter
                    if $rateIter == 0
                        set rate 1
                    else
                        set rate $rateIter * 5
                    endif
                    setrate $rate
                    log Rate $rate
                    set dirname $volname-$l-$tname-$lit-$rate-n-0
                    repeat 5 as transform
                        settranslation 0 0 0
                        resetrotation
                        if $volume == 0
                            addrotationx -90
                            addrotationy 180
                            setuseortho true
                        endif
                        if $volume == 1
                            addrotationx -90
                            addrotationy -90
                            setuseortho true
                        endif
                        if $volume == 2
                            settransformparams -0.202214763_0.978688002_-0.0357658304_0_-0.202166244_-0.00598103832_0.979332626_0_0.958247483_0.205267921_0.199067056_0_0_0_0_1~0_0_1.50000024
                            setuseortho false
                        endif
                        if $volume == 3
                            settransformparams -0.969087362_0.246711731_0.00148320443_0_-0.012833355_-0.0564099029_0.998324871_0_0.246383488_0.96744591_0.0578331985_0_0_0_0_1~0_0_0.800000072
                            setuseortho false
                        endif
                        if $volume == 4
                            addrotationy 90
                            setuseortho true
                        endif
                        if $volume == 5
                            addrotationy 90
                            setuseortho true
                        endif
                        if $volume == 6
                            settransformparams -0.94459784_-0.268845826_-0.18829757_0_-0.200813919_0.927143097_-0.316352993_0_0.259629518_-0.261014193_-0.929764926_0_0_0_0_1~0_0_0.100000016
                            setuseortho false
                        endif
                        if $volume == 7
                            settransformparams 0.84378463_-0.535941601_0.0281707458_0_0.262970418_0.458637923_0.848821521_0_-0.467838824_-0.708814919_0.527927518_0_0_0_0_1~0_0_0
                            setuseortho false
                        endif
                        if $volume == 8
                            settransformparams 0.84378463_-0.535941601_0.0281707458_0_0.262970418_0.458637923_0.848821521_0_-0.467838824_-0.708814919_0.527927518_0_0_0_0_1~0_0_0
                            setuseortho false
                        endif
                        if $transform == 0
                            setdir output/rotationy/$dirname
                            repeat 120 as r
                                screenshot $dirname-$r.$fileformat
                                addrotationy 3
                            endrepeat
                            setdir output
                        endif
                        if $transform == 1
                            setdir output/rotationdiagonal/$dirname
                            repeat 120 as r
                                screenshot $dirname-$r.$fileformat
                                addrotationaxis 0.57735027 0.57735027 0.57735027 3
                            endrepeat
                            setdir output
                        endif
                        if $transform == 2
                            setdir output/translationx/$dirname
                            addtranslation -0.5 0 0
                            repeat 100 as r
                                screenshot $dirname-$r.$fileformat
                                addtranslation 0.01 0 0
                            endrepeat
                            setdir output
                        endif
                        if $transform == 3
                            setdir output/translationy/$dirname
                            addtranslation 0 0.5 0
                            repeat 100 as r
                                screenshot $dirname-$r.$fileformat
                                addtranslation 0 -0.01 0
                            endrepeat
                            setdir output
                        endif
                        if $transform == 4
                            setdir output/zoom/$dirname
                            setuseortho false
                            addtranslation 0 0 0.5
                            repeat 100 as r
                                screenshot $dirname-$r.$fileformat
                                addtranslation 0 0 -0.01
                            endrepeat
                            setdir output
                        endif
                    endrepeat
                endrepeat
                setrate 1
                if $lighting == 0
                    set m $m + 3
                else
                    set m $m + 2
                endif
                repeat 4 as vmethod
                    set vm $m + $vmethod
                    setmethod $vm
                    log Virtual Method $vm
                    if $vmethod == 0
                        set vname linvs
                    endif
                    if $vmethod == 1
                        set vname crvs
                    endif
                    if $vmethod == 2
                        set vname hermvs
                    endif
                    if $vmethod == 3
                        set vname monhermvs
                    endif
                    repeat 20 as vs
                        set subdiv $vs + 1
                        setsubdiv $subdiv
                        log Subdiv $subdiv
                        set dirname $volname-$l-$tname-$lit-1-$vname-$subdiv
                        repeat 5 as transform
                            settranslation 0 0 0
                            resetrotation
                            if $volume == 0
                                addrotationx -90
                                addrotationy 180
                                setuseortho true
                            endif
                            if $volume == 1
                                addrotationx -90
                                addrotationy -90
                                setuseortho true
                            endif
                            if $volume == 2
                                settransformparams -0.202214763_0.978688002_-0.0357658304_0_-0.202166244_-0.00598103832_0.979332626_0_0.958247483_0.205267921_0.199067056_0_0_0_0_1~0_0_1.50000024
                                setuseortho false
                            endif
                            if $volume == 3
                                settransformparams -0.969087362_0.246711731_0.00148320443_0_-0.012833355_-0.0564099029_0.998324871_0_0.246383488_0.96744591_0.0578331985_0_0_0_0_1~0_0_0.800000072
                                setuseortho false
                            endif
                            if $volume == 4
                                addrotationy 90
                                setuseortho true
                            endif
                            if $volume == 5
                                addrotationy 90
                                setuseortho true
                            endif
                            if $volume == 6
                                settransformparams -0.94459784_-0.268845826_-0.18829757_0_-0.200813919_0.927143097_-0.316352993_0_0.259629518_-0.261014193_-0.929764926_0_0_0_0_1~0_0_0.100000016
                                setuseortho false
                            endif
                            if $volume == 7
                                settransformparams 0.84378463_-0.535941601_0.0281707458_0_0.262970418_0.458637923_0.848821521_0_-0.467838824_-0.708814919_0.527927518_0_0_0_0_1~0_0_0
                                setuseortho false
                            endif
                            if $volume == 8
                                settransformparams 0.84378463_-0.535941601_0.0281707458_0_0.262970418_0.458637923_0.848821521_0_-0.467838824_-0.708814919_0.527927518_0_0_0_0_1~0_0_0
                                setuseortho false
                            endif
                            if $transform == 0
                                setdir output/rotationy/$dirname
                                repeat 120 as r
                                    screenshot $dirname-$r.$fileformat
                                    addrotationy 3
                                endrepeat
                                setdir output
                            endif
                            if $transform == 1
                                setdir output/rotationdiagonal/$dirname
                                repeat 120 as r
                                    screenshot $dirname-$r.$fileformat
                                    addrotationaxis 0.57735027 0.57735027 0.57735027 3
                                endrepeat
                                setdir output
                            endif
                            if $transform == 2
                                setdir output/translationx/$dirname
                                addtranslation -0.5 0 0
                                repeat 100 as r
                                    screenshot $dirname-$r.$fileformat
                                    addtranslation 0.01 0 0
                                endrepeat
                                setdir output
                            endif
                            if $transform == 3
                                setdir output/translationy/$dirname
                                addtranslation 0 0.5 0
                                repeat 100 as r
                                    screenshot $dirname-$r.$fileformat
                                    addtranslation 0 -0.01 0
                                endrepeat
                                setdir output
                            endif
                            if $transform == 4
                                setdir output/zoom/$dirname
                                setuseortho false
                                addtranslation 0 0 0.5
                                repeat 100 as r
                                    screenshot $dirname-$r.$fileformat
                                    addtranslation 0 0 -0.01
                                endrepeat
                                setdir output
                            endif
                        endrepeat
                    endrepeat
                endrepeat
            endrepeat
        endrepeat
    endrepeat
endrepeat