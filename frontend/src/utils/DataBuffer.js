// src/utils/DataBuffer.js
export class DataBuffer {
    constructor(size) {
        this.size = size;
        this.buffer = new Float32Array(size);
        this.position = 0;
        this.full = false;
    }

    add(value) {
        this.buffer[this.position] = value;
        this.position = (this.position + 1) % this.size;
        if (this.position === 0) this.full = true;
    }

    isFull() {
        return this.full;
    }

    getData() {
        if (!this.full && this.position === 0) return null;

        // Arrange data so most recent samples are at the end
        const result = new Float32Array(this.size);
        if (this.full) {
            result.set(this.buffer.slice(this.position));
            result.set(this.buffer.slice(0, this.position), this.size - this.position);
        } else {
            result.set(this.buffer.slice(0, this.position));
        }
        return result;
    }
}