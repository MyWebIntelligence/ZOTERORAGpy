/**
 * RAGpy - Pagination Component
 * Reusable pagination controls for lists and tables
 */

/**
 * Renders pagination controls HTML
 * @param {Object} config - Pagination configuration
 * @param {number} config.currentPage - Current page number (1-indexed)
 * @param {number} config.totalPages - Total number of pages
 * @param {number} config.total - Total items count
 * @param {number} config.perPage - Items per page
 * @param {string} [config.onPageChange] - Name of function to call on page change (defaults to 'changePage')
 * @returns {string} HTML string for pagination controls
 */
function renderPagination({ currentPage, totalPages, total, perPage, onPageChange = 'changePage' }) {
    // Don't show pagination if only one page or no items
    if (totalPages <= 1 || total === 0) {
        return '';
    }

    // Calculate display range
    const start = ((currentPage - 1) * perPage) + 1;
    const end = Math.min(currentPage * perPage, total);

    let html = '<div class="pagination">';

    // Info text: "Affichage 1-10 sur 45"
    html += `<div class="pagination-info">Affichage ${start}-${end} sur ${total}</div>`;

    html += '<div class="pagination-controls">';

    // Previous button
    const prevDisabled = currentPage === 1 ? 'disabled' : '';
    html += `<button class="pagination-btn" ${prevDisabled} onclick="${onPageChange}(${currentPage - 1})" title="Page précédente">
        <i class="bi bi-chevron-left"></i>
    </button>`;

    // Page numbers with smart ellipsis
    const pageNumbers = getPageNumbers(currentPage, totalPages);
    pageNumbers.forEach(page => {
        if (page === '...') {
            html += '<span class="pagination-ellipsis">...</span>';
        } else {
            const activeClass = page === currentPage ? 'active' : '';
            html += `<button class="pagination-btn ${activeClass}"
                             onclick="${onPageChange}(${page})"
                             title="Page ${page}">${page}</button>`;
        }
    });

    // Next button
    const nextDisabled = currentPage === totalPages ? 'disabled' : '';
    html += `<button class="pagination-btn" ${nextDisabled} onclick="${onPageChange}(${currentPage + 1})" title="Page suivante">
        <i class="bi bi-chevron-right"></i>
    </button>`;

    html += '</div></div>';

    return html;
}

/**
 * Generate smart page number array with ellipsis
 * Algorithm shows: 1 ... 4 5 [6] 7 8 ... 20
 *
 * Logic:
 * - Always show first and last page
 * - Show current page ± 1 page
 * - Use ellipsis (...) for gaps
 *
 * @param {number} current - Current page number
 * @param {number} total - Total number of pages
 * @returns {Array<number|string>} Array of page numbers and ellipsis
 */
function getPageNumbers(current, total) {
    // If 7 or fewer pages, show all
    if (total <= 7) {
        return Array.from({ length: total }, (_, i) => i + 1);
    }

    const pages = [];

    // Always show first page
    pages.push(1);

    // Calculate range around current page
    const start = Math.max(2, current - 1);
    const end = Math.min(total - 1, current + 1);

    // Add left ellipsis if needed (gap between 1 and start)
    if (start > 2) {
        pages.push('...');
    }

    // Add middle pages (current ± 1)
    for (let i = start; i <= end; i++) {
        pages.push(i);
    }

    // Add right ellipsis if needed (gap between end and last)
    if (end < total - 1) {
        pages.push('...');
    }

    // Always show last page
    if (total > 1) {
        pages.push(total);
    }

    return pages;
}

/**
 * Example usage:
 *
 * const paginationHTML = renderPagination({
 *     currentPage: 3,
 *     totalPages: 10,
 *     total: 95,
 *     perPage: 10,
 *     onPageChange: 'loadProjectSessions'
 * });
 *
 * document.getElementById('pagination-container').innerHTML = paginationHTML;
 *
 * function loadProjectSessions(page) {
 *     // Fetch data for page...
 * }
 */
