'use client';

import { useEffect, useRef } from 'react';

import { ErrorNotice } from '../lib/errors';

export function ErrorModal({ close, notice }: { close: () => void; notice: ErrorNotice }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
    return () => { if (dialog?.open) dialog.close(); };
  }, []);
  function retry() { close(); notice.retry?.(); }
  return <dialog aria-describedby="error-description" aria-labelledby="error-title" className="error-modal" onCancel={close} onClick={(event) => { if (event.target === event.currentTarget) close(); }} ref={dialogRef} role="alertdialog">
    <div className="error-card">
      <button aria-label="Close error message" className="error-close" onClick={close} type="button">×</button>
      <span className="error-symbol" aria-hidden="true">!</span>
      <div className="error-copy"><p>Unable to continue</p><h2 id="error-title">{notice.title}</h2><p id="error-description">{notice.body}</p>{notice.code && <small>Reference: {notice.code}</small>}</div>
      <div className="error-actions"><button className="secondary" onClick={close} type="button">Close</button>{notice.retry && <button autoFocus onClick={retry} type="button">Try again</button>}</div>
    </div>
  </dialog>;
}
