# Impedance Material Enhancement

A small notebook-based project for improving the readability of scanned handwritten PDFs.

## Structure

impedance_material/
├── .venv/
├── input/
│   └── original/
├── output/
│   └── enhanced/
├── notebooks/
│   ├── 01_tests.ipynb
│   └── 02_process_pdf.ipynb
├── requirements.txt
├── README.md
└── .gitignore

- input/original/: original PDFs, kept unchanged
- output/enhanced/: processed PDFs
- notebooks/01_tests.ipynb: interactive parameter testing
- notebooks/02_process_pdf.ipynb: final PDF processing

## Setup

`sirius activate mamba`
`source .venv/bin/activate`
`code ~/coding/impedance_material`

In VS Code, select .venv/bin/python as the notebook kernel.

## Goal

The enhancement workflow may include:

- deskewing
- background normalization
- contrast enhancement
- light sharpening

Each PDF page is processed independently, so pages may have different dimensions.