# Seedcraft

**Deterministic, correlation-aware synthetic data generation.**

Seedcraft generates realistic synthetic datasets where fields are meaningfully correlated, not just independently random. Unlike Faker or other synthetic data tools that produce each column in isolation, Seedcraft lets you define inter-field relationships — age influences income, department determines bonus structure, temperature inversely affects humidity — and generates coherent data that respects those correlations every time.

## Why Seedcraft?

Most synthetic data tools generate independent random values per field. But real data has **structure**: a 22-year-old intern doesn't earn $500K, grocery transactions aren't $10,000, and high humidity doesn't pair with scorching temperatures. When testing with independently-random data, you miss the bugs that hide in **data relationships**.

Seedcraft fills this gap:

- **Correlated fields**: Linear, monotonic, conditional, derived, and mutual-exclusion correlations
- **Deterministic**: Same seeds always produce the same data, across machines and runs
- **Field-independent streams**: Adding a new field doesn't change existing fields' values
- **Distribution-aware**: Normal, lognormal, exponential, beta, Poisson, Zipf, and uniform distributions
- **Zero dependencies**: Pure Python 3.10+, no external packages required
- **Built-in analysis**: Verify that generated correlations actually hold with statistical checks

## Installation

```bash
# Clone and use directly — no pip install needed
git clone <repo-url>
cd seedcraft

# Or install in development mode
pip install -e .
```

**Requirements:** Python 3.10+ (standard library only, no external dependencies)

## Quick Start

```python
from src import SeedEngine, DataGenerator, Schema, Field, Correlation, Distribution

# Define a schema with correlated fields
schema = (
    Schema("employees", row_count=1000)
    .add_field(Field.string_id("emp_id", prefix="EMP"))
    .add_field(Field.integer("years_experience", low=0, high=40))
    .add_field(Field.real("salary", min_val=25000, max_val=500000))
    .add_field(Field.choice("department", ["engineering", "sales", "hr"]))
)

# Experience drives salary
schema.add_correlation(
    Correlation.linear("years_experience", "salary",
                       strength=0.8, slope=5000, intercept=35000)
)

# Generate with deterministic seeds
engine = SeedEngine([3813, 7889, 6140, 439])
data = DataGenerator(engine).generate(schema)

# Export
from src.exporters import to_csv, to_json
print(to_csv(data))
```

## Correlation Types

### Linear
Target tracks source linearly: `y ≈ strength * (slope * x + intercept) + (1 - strength) * noise`

```python
Correlation.linear("age", "income", strength=0.8, slope=800, intercept=20000)
```

### Conditional
Target distribution depends on source category:

```python
Correlation.conditional("department", "bonus_pct", {
    "engineering": Distribution.normal(12, 3),
    "sales": Distribution.normal(25, 8),
    "hr": Distribution.normal(8, 2),
})
```

### Derived
Target is a deterministic function of source:

```python
Correlation.derived("subtotal", "tax", func=lambda x: round(x * 0.085, 2))
```

### Monotonic
Target increases/decreases with source, with controllable noise:

```python
Correlation.monotonic("study_hours", "gpa", strength=0.75, increasing=True)
```

### Mutual Exclusive
When source is True, target is always False:

```python
Correlation.mutual_exclusive("is_premium", "free_shipping")
```

## Distributions

All numeric fields support configurable distributions:

| Distribution | Constructor | Use Case |
|---|---|---|
| Uniform | `Distribution.uniform(0, 100)` | Equal probability across range |
| Normal | `Distribution.normal(50, 10)` | Bell curve (ages, measurements) |
| Lognormal | `Distribution.lognormal(10, 0.5)` | Right-skewed (incomes, prices) |
| Exponential | `Distribution.exponential(5)` | Time between events, counts |
| Beta | `Distribution.beta(2, 5)` | Bounded [0,1] (percentages, rates) |
| Poisson | `Distribution.poisson(3)` | Count data (orders per day) |
| Zipf | `Distribution.zipf(2)` | Power law (popularity, word frequency) |

## Preset Schemas

Four ready-to-use schemas for common scenarios:

```bash
# List available presets
python -m src.cli list

# Generate e-commerce customer data
python -m src.cli generate ecommerce_customers -n 500 -f csv -o customers.csv

# Analyze sensor readings with correlation matrix
python -m src.cli analyze sensor_readings -n 1000
```

| Preset | Fields | Correlations | Domain |
|---|---|---|---|
| `ecommerce_customers` | 10 | age→income, income→spending, tier→discount, premium⊕shipping | E-commerce |
| `sensor_readings` | 8 | temp→humidity (inverse), temp→pressure, light→temp | IoT / sensors |
| `student_records` | 8 | study_hours→gpa, extracurriculars→leadership, gpa→scholarship | Education |
| `financial_transactions` | 9 | account_age→limit, category→amount | Finance / fraud |

## CLI Usage

```bash
# Generate data
python -m src.cli generate <preset> [-n ROWS] [-s SEED...] [-f csv|json] [-o FILE]

# Analyze generated data (stats + correlation matrix)
python -m src.cli analyze <preset> [-n ROWS] [-s SEED...]

# List presets
python -m src.cli list

# Show schema details
python -m src.cli info <preset>
```

Custom seeds for reproducibility:
```bash
python -m src.cli generate sensor_readings -n 100 -s 42 99 7 -f json
```

## Statistical Verification

Seedcraft includes built-in tools to verify that correlations actually hold:

```python
from src.analyzer import pearson_correlation, correlation_matrix, verify_correlations

# Check pairwise correlations
matrix = correlation_matrix(data)

# Verify specific expected correlations
results = verify_correlations(data, [
    ("temperature", "humidity", -0.5),
    ("age", "income", 0.6),
], tolerance=0.25)

for r in results:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"[{status}] {r['source']} -> {r['target']}: {r['actual']}")
```

## Architecture

```
src/
├── __init__.py          # Public API
├── schema.py            # Field, Distribution, Correlation, Schema definitions
├── engine.py            # SeedEngine: deterministic multi-stream RNG
├── generator.py         # DataGenerator: correlation-aware generation
├── analyzer.py          # Statistical analysis and verification
├── exporters.py         # CSV, JSON export
├── presets.py           # Ready-to-use schema templates
└── cli.py               # Command-line interface

tests/                   # 128 tests covering all modules
demo/
└── demo.py              # Interactive demonstration
```

### Key Design Decisions

1. **Hash-based stream splitting**: Each field gets an independent Random stream derived by hashing `(master_seed, field_name)`. This means adding/removing fields never changes other fields' output — a critical property for evolving schemas.

2. **Topological generation order**: Fields are generated in dependency order via topological sort. Independent fields first, then correlated fields using their source values.

3. **Correlation as modulation**: Correlated fields blend a signal component (derived from source) with a noise component (from the target's own distribution), controlled by the `strength` parameter.

4. **Zero external dependencies**: The entire library uses only Python's standard library (`random`, `hashlib`, `math`, `csv`, `json`), making it trivially portable.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Running the Demo

```bash
python demo/demo.py
```

The demo showcases: basic generation, correlated fields, all preset schemas, statistical analysis with correlation matrices, deterministic reproducibility, derived fields, and export formats.

## License

MIT
