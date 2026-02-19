import great_expectations as gx
import pandas as pd

context = gx.get_context()

# Load data
df = pd.read_csv('data/raw/customer_data.csv')

# Create expectations
validator = context.sources.pandas_default.read_dataframe(df)

# Define rules
validator.expect_column_values_to_be_between('monthly_charges', min_value=0, max_value=200)
validator.expect_column_values_to_not_be_null('customer_id')
validator.expect_column_values_to_be_in_set('contract_type', ['Month-to-Month', 'One Year', 'Two Year'])
validator.expect_table_row_count_to_be_between(min_value=10000, max_value=100000)

# Run validation
result = validator.validate()

if result.success:
    print("✓ Data validation passed")
else:
    print("✗ Data validation failed")
    print(result)
    exit(1)
