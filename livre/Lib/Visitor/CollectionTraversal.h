
/* Copyright (c) 2011-2014, EPFL/Blue Brain Project
 *                          Ahmet.Bilgili@epfl.ch
 *
 * This file is part of Livre <https://github.com/BlueBrain/Livre>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _CollectionTraversal_h_
#define _CollectionTraversal_h_

#include <livre/core/types.h>
#include <livre/core/Visitor/RenderNodeVisitor.h>
#include <livre/core/Visitor/VisitState.h>
#include <livre/core/dashTypes.h>

#include <boost/type_traits.hpp>
#include <algorithm>

namespace livre
{
/**
 * The CollectionTraversal class is used to traverse the DashNodeVector with a given visitor.
 */
class CollectionTraversal
{
public:

    CollectionTraversal() {}

    /**
     * Traverse a collection with the given visitor. Visitor can decide to end traversal.
     * @param dashNodeVector the (backward-)iterable dash node collection.
     * @param visitor Visitor object.
     * @param reverse If reverse given, the dash vector is traversed in reverse (default: false)
     * @return True if traversal is completed, without being completed.
     */
    bool traverse( DashNodeVector& dashNodeVector,
                   RenderNodeVisitor& visitor,
                   const bool reverse = false )
    {
        DashNodeVector::const_iterator begin;
        DashNodeVector::const_iterator end;

        if( reverse )
        {
            begin = dashNodeVector.end();
            end = dashNodeVector.begin();
        }
        else
        {
            begin = dashNodeVector.begin();
            end = dashNodeVector.end();
        }

        VisitState state;
        visitor.onTraverseBegin( state );

        if( !state.getBreakTraversal() )
        {
            for( DashNodeVector::const_iterator i = begin; i != end;
                 reverse ? --i : ++i )
            {
                const dash::NodePtr& node = *i;
                DashRenderNode renderNode( node );
                visitor.visit( renderNode, state );
                if( state.getBreakTraversal( ) )
                    break;
            }
        }

        visitor.onTraverseEnd( state );
        return state.getBreakTraversal();
    }

};

}

#endif // _CollectionTraversal_h_
