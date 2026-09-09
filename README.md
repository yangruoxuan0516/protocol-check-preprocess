# to get protocol.jsonl / checklist.jsonl / enriched_checklist.jsonl,
# run the following 3 scripts marked with *xxx*, with proper paths

*extract_protocol.py*
python extract_protocol.py "xxx.docx" -o "xxx.jsonl"

*extract_checklist.py*
python extract_checklist.py "xxx.xlsx" -o "xxx.jsonl"
# before running the script, correct the raw excel as described below
- correctness checklist, 4E, 拼写错误，应改为 Applicability
- completeness checklist，48D，格式错误，应改为 5.4.3 d
- ARP4754A TABLE 暂不支持，因为从pdf extract table比较复杂，而且 checklist 只出现 2 次，都是 "TABLE A-1 objective 4.4"


no output path is required for the following 3 scripts, as they will not be run alone, but mainly serve as libraries for the enrich script
> extract_arp.py
pip install pymupdf
python extract_arp.py "xxx.pdf" "ARP4754 5.4.4.1 a" --inspect-margins

> extract_do.py
python extract_do.py "xxx.pdf" "5.3 e" --inspect-margins

> extract_std.py
python extract_std.py "xxx.pdf" "RS36" --inspect-margins



*enrich_checklist_with_reference.py*
python enrich_checklist_with_reference.py "checklist.jsonl" --arp4754-pdf "xxx.pdf" --std-pdf "xxx.pdf" --do297-pdf "xxx.pdf" -o "enriched_checklist.jsonl"


*extract_arinc.py*
python extract_arinc.py "xxx.pdf" --inspect-margins

python extract_arinc.py "xxx.pdf" --inspect-sections

python extract_arinc.py ARINC664P2.pdf --document-id ARINC664P2 -o arinc664p2_chunks.jsonl

python extract_arinc.py ARINC664P7.pdf --document-id ARINC664P7 -o arinc664p7_chunks.jsonl