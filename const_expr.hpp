#pragma once

constexpr unsigned bitsNeeded(unsigned n) {
	return n <= 1 ? 0 : 1 + bitsNeeded((n + 1) / 2);
}
