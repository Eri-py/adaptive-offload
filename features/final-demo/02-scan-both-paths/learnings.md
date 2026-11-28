
## Task 1
- `expo install` resolved expo-image-picker ~57.0.22 and expo-image-manipulator ~57.0.23 (SDK 57 line).
- expo-file-system 57 `File` API: `new File(Paths.document, name)`, `.exists`, `.create()`, `.write(string)`, `await .text()`. No legacy readAsStringAsync needed.
- ATS: `NSAllowsArbitraryLoads` chosen because per-domain `NSExceptionDomains` entries match hostnames, not IP literals/CIDR ranges, so they cannot cover 100.64.0.0/10. Broad but fine for a course demo.
- Requires one new EAS development build (new native modules + Info.plist changes).
