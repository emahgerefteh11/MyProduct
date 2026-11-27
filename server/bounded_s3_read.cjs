'use strict';

const {addAbortSignal} = require('node:stream');

class CurObjectReadError extends Error {
    constructor(code, message) {
        super(message);
        this.code = code;
    }
}

const tooLarge = () => new CurObjectReadError('OBJECT_TOO_LARGE', 'CUR object exceeds the byte limit.');

// getObject is supplied by an already scoped server-side client. Do not use
// the SDK's buffer conversion helper: the limit must apply as chunks arrive.
async function readBoundedS3Object({getObject, key, maxBytes, signal, ifMatch}) {
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new TypeError('A positive byte limit is required.');
    signal?.throwIfAborted();

    let result;
    try {
        result = await getObject({key, signal, ifMatch});
    } catch (error) {
        if (ifMatch && error?.$metadata?.httpStatusCode === 412) {
            throw new CurObjectReadError('MANIFEST_CHANGED', 'The current CUR revision changed; retry the refresh.');
        }
        throw error;
    }

    const body = result?.Body;
    try {
        signal?.throwIfAborted();
        if (Number.isFinite(result?.ContentLength) && result.ContentLength > maxBytes) throw tooLarge();
        if (typeof body?.[Symbol.asyncIterator] !== 'function' || typeof body?.destroy !== 'function') {
            throw new CurObjectReadError('INVALID_BODY', 'An S3 Node body stream is required.');
        }
        if (signal) addAbortSignal(signal, body);
        const chunks = [];
        let seen = 0;
        for await (const chunk of body) {
            if (!(chunk instanceof Uint8Array)) {
                throw new CurObjectReadError('INVALID_BODY', 'The CUR object must contain bytes.');
            }
            seen += chunk.byteLength;
            if (seen > maxBytes) throw tooLarge();
            chunks.push(chunk);
        }
        signal?.throwIfAborted();
        if (Number.isSafeInteger(result.ContentLength) && seen !== result.ContentLength) {
            throw new CurObjectReadError('INCOMPLETE_BODY', 'The S3 response ended before the expected object length.');
        }
        return {bytes: Buffer.concat(chunks, seen), etag: result.ETag ?? null};
    } finally {
        // Also close on an oversized ContentLength or cancellation before iteration.
        if (typeof body?.destroy === 'function' && !body.destroyed) body.destroy();
    }
}

module.exports = {readBoundedS3Object};
