# V3: Analysis-Focused Design

## Philosophy Change

**V2:** Image and analysis balanced
**V3:** Analysis is primary, image is reference

## Key Changes

### 📐 Layout Rebalance

**Image Size:**
- V2: 60% of canvas
- V3: 50% of canvas ✅
- Reason: Viewer doesn't need to see every pixel, just get the analysis

**Panel Size:**
- V2: 40% of canvas
- V3: 50% of canvas ✅
- Reason: Analysis panels are the actual value

### 🎨 Color Analysis - Now the Star

**V2:**
- 30% of panel space
- 6 colors
- Basic percentage labels

**V3:**
- **50% of panel space** ✅
- 8 colors (more detail)
- **Large percentage labels**
- Color temperature indicators (warm/cool)
- Sorted by dominance
- Minimum 35px per bar (very readable)
- Gradient 3D effect on bars

### 📊 Frequency Analysis - Actually Useful Now

**V2 (Radial FFT):**
```
      ○○○
    ○    ○
   ○  ●   ○
    ○    ○
      ○○○
```
- Pretty but not informative
- Hard to read actual values
- Decorative

**V3 (Spectral Energy Graph):**
```
Energy
  │      ╱╲
  │     ╱  ╲    ╱╲
  │  ╱╲╱    ╲  ╱  ╲
  │╱          ╲╱    ╲
  └──────────────────→
  Low          High
  Frequency
```
- **Area chart** showing energy distribution
- Clear axes and labels
- Red dots highlight significant peaks
- Shows low vs high frequency energy
- **Actually informative!** ✅

### 🔍 Reduced Image Overlays

**V2:**
- 20 contour levels
- 30px vector spacing
- 200 features
- Full opacity

**V3:**
- **15 contour levels** (25% fewer)
- **40px vector spacing** (sparser)
- **150 features** (25% fewer)
- **Reduced opacity** (90% instead of 100%)

Reason: Image is reference, overlays shouldn't overwhelm

## Panel Space Allocation (Vertical Layout)

**V2:**
```
┌─────────────┐
│ FFT   (40%) │
├─────────────┤
│ COLOR (30%) │
├─────────────┤
│ STATS (30%) │
└─────────────┘
```

**V3:**
```
┌─────────────┐
│ COLOR (50%) │ ← LARGEST!
├─────────────┤
│ FREQ  (35%) │ ← Informative
├─────────────┤
│ STATS (15%) │ ← Summary
└─────────────┘
```

## New Features in V3

### Color Temperature
Each color gets a temperature classification:
- **Warm** (red dot) - Red-dominant colors
- **Cool** (blue dot) - Blue-dominant colors
- **Neutral** (gray dot) - Balanced colors

### Frequency Statistics
- **Low frequency energy** - Large shapes/structure
- **High frequency energy** - Details/texture
- **Peak detection** - Dominant frequencies highlighted

### Better Color Sorting
- **Sorted by percentage** (most common first)
- Was sorted by luminance in V2
- More intuitive ordering

## Visual Style

### Cleaner Image Presentation
- Minimal grid (8 lines instead of 16)
- Lighter overlays (don't compete with panels)
- Simple brackets
- Image is a "reference preview"

### Emphasis on Data
- Larger fonts for percentages
- Clear graph axes
- More spacing in panels
- Professional analytics look

## Use Cases

**V2 Best For:**
- Showcasing the photo with analysis
- Visual appeal over data
- Social media sharing

**V3 Best For:**
- Understanding image composition ✅
- Color palette extraction ✅
- Frequency/texture analysis ✅
- Professional/research use ✅

## Configuration Differences

```python
# V2
PANEL_SIZE_LANDSCAPE = 0.40  # 40%
NUM_COLORS = 6
CONTOUR_LEVELS = 20
VECTOR_GRID_SPACING = 30

# V3
PANEL_SIZE_LANDSCAPE = 0.50  # 50% - panels dominate!
NUM_COLORS = 8               # More detail
CONTOUR_LEVELS = 15          # Fewer (image is reference)
VECTOR_GRID_SPACING = 40     # Sparser
BASE_IMAGE_FADE = 0.08       # Less fade (subtle)
OVERLAY_ALPHA_BOOST = 0.9    # Reduced overlay intensity
```

## Usage

```bash
# Analysis-focused output
python orbital_survey_v3.py input.jpg

# Output focuses on:
# - What colors dominate? (50% of panel)
# - How is frequency energy distributed? (35% of panel)
# - What are the key stats? (15% of panel)
```

## Recommendation

Use V3 when:
- ✅ You want to **analyze** the image
- ✅ Color composition is important
- ✅ You need actual data/metrics
- ✅ Professional/research context

Use V2 when:
- Photo itself should be prominent
- Visual appeal > analytical data
- Sharing on social media
