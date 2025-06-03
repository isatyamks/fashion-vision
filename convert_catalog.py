import pandas as pd

excel_file = 'product_data.xlsx'
csv_file = 'catalog.csv'

try:
    # Read the Excel file
    df = pd.read_excel(excel_file)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

    print(f"Successfully converted {excel_file} to {csv_file}")

except FileNotFoundError:
    print(f"Error: {excel_file} not found.")
except Exception as e:
    print(f"An error occurred during conversion: {e}") 