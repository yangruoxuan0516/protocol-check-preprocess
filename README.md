*extract_protocol.py*
python extract_protocol.py "xxx.docx" --output-dir "xxx"

*extract_checklist.py*
python extract_checklist.py "xxx.xlsx" -o "xxx.jsonl"

*extract_arp.py*
pip install pymupdf
python extract_arp.py "xxx.pdf" "ARP4754 5.4.4.1 a" --inspect-margins
no output path is required as this script will not be run alone, it mainly serves as a library for the enrich script

*extract_do.py*
python extract_do.py "xxx.pdf" "5.3 e" --inspect-margins

*extract_std.py*
python extract_std.py "xxx.pdf" "RS36" --inspect-margins

*enrich_checklist_with_reference.py*
python enrich_checklist_with_reference.py "checklist.jsonl" --arp4754-pdf "xxx.pdf" --std-pdf "xxx.pdf" --do297-pdf "xxx.pdf" -o "enriched_checklist.jsonl"

