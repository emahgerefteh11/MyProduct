'use strict';

// Bind an already authorized S3 GetObject client to a single source and
// configured billing-period manifest. The connector still checks the exact
// manifest-listed parts before calling this reader; IAM is a separate fence.
function bindKeptCurS3Reader({tenant, source, bucket, expectedBucketOwner, manifestKey, sendGetObject}) {
    const slash = typeof manifestKey === 'string' ? manifestKey.lastIndexOf('/') : -1;
    const directory = slash > 0 ? manifestKey.slice(0, slash) : '';
    const match = slash > 0 && /^(.+)-Manifest\.json$/.exec(manifestKey.slice(slash + 1));
    if (!tenant || !source || typeof sendGetObject !== 'function' ||
        typeof bucket !== 'string' || typeof expectedBucketOwner !== 'string' ||
        !/^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/.test(bucket) || !/^\d{12}$/.test(expectedBucketOwner) ||
        !match || !/\/\d{8}-\d{8}$/.test(directory)) {
        throw new TypeError('An approved CUR source, bucket owner and exact period manifest are required.');
    }

    const prefix = directory + '/';
    const stem = match[1] + '-';
    function permittedPart(key) {
        if (!key.startsWith(prefix)) return false;
        const segments = key.slice(prefix.length).split('/');
        if (segments.length !== 2 || !/^\d{8}T\d{6}Z$/.test(segments[0]) || !segments[1].startsWith(stem)) return false;
        return /^[a-zA-Z0-9_.-]+\.csv\.gz$/.test(segments[1].slice(stem.length));
    }

    return async function read({tenant: callerTenant, source: callerSource, key, ifMatch, signal}) {
        if (callerTenant !== tenant || callerSource !== source || typeof key !== 'string' ||
            (key !== manifestKey && !permittedPart(key)) ||
            (ifMatch !== undefined && (key !== manifestKey || typeof ifMatch !== 'string' ||
                !ifMatch || ifMatch === '*' || ifMatch.length > 256 || /[\r\n]/.test(ifMatch)))) {
            throw new Error('CUR object is outside the configured source scope.');
        }
        signal?.throwIfAborted();
        const input = {Bucket: bucket, Key: key, ExpectedBucketOwner: expectedBucketOwner};
        if (ifMatch !== undefined) input.IfMatch = ifMatch;
        return sendGetObject(input, {abortSignal: signal});
    };
}

module.exports = {bindKeptCurS3Reader};
