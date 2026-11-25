import React from 'react';
import { Metadata } from 'next';
import { Form4PilotEvidencePreview } from '../../../components/pilot/Form4PilotEvidencePreview';

export const metadata: Metadata = {
  title: 'Form 4 Cohort Pilot Evidence (Private Preview) | ARX Terminal',
  description: 'Private operator preview of SEC Form 4 cohort pilot reconciliation receipts for Issue #165.',
};

export default function Form4PilotEvidencePage() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto">
      {/* Breadcrumb / Context Header */}
      <div className="mb-6 flex flex-wrap items-center gap-2 text-xs font-mono text-slate-400">
        <span className="text-slate-500">ARX Terminal</span>
        <span>/</span>
        <span className="text-slate-500">Operator Staging</span>
        <span>/</span>
        <span className="text-slate-500">Issue #165</span>
        <span>/</span>
        <span className="text-purple-400 font-semibold">Form 4 Pilot Evidence Preview</span>
      </div>

      <Form4PilotEvidencePreview />
    </main>
  );
}
